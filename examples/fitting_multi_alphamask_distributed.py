#!/usr/bin/env python

import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor, optim

# Required libraries:
#   pip install gsplat twodgs lpips pytorch-msssim
from gsplat import rasterization
from twodgs import TwoDGaussians

# Additional metrics
import lpips
from pytorch_msssim import ssim as ssim_fn

def calculate_psnr(img: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate PSNR between img and gt."""
    mse = np.mean((img - gt) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def calculate_lpips(img: np.ndarray, gt: np.ndarray, device: torch.device = torch.device('cpu')) -> float:
    """Calculate LPIPS between img and gt in [0..1], shape [H,W,3]."""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Initialize model on CPU first
        loss_fn_vgg = lpips.LPIPS(net='vgg')
        # Move to specified device only when needed
        loss_fn_vgg = loss_fn_vgg.to(device)
        
        # LPIPS expects inputs in [-1, +1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2.0 - 1.0
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = loss_fn_vgg(img_t, gt_t).item()
        
        # Clean up
        loss_fn_vgg.cpu()
        del loss_fn_vgg
        del img_t
        del gt_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return lpips_val
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # If OOM occurs, try on CPU
            print("Warning: CUDA OOM, calculating LPIPS on CPU")
            return calculate_lpips(img, gt, device=torch.device('cpu'))
        else:
            raise e

def calculate_ssim(img: np.ndarray, gt: np.ndarray, device: torch.device = torch.device('cpu')) -> float:
    """Calculate SSIM between img and gt in [0..1], shape [H,W,3]."""
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        ssim_val = ssim_fn(img_t, gt_t, data_range=1.0, size_average=True)
    return ssim_val.item()

@dataclass
class TrainArgs:
    height: int = 256
    width: int = 256
    num_points: int = 200
    save_imgs: bool = True
    img_path: Optional[Path] = None
    image_folder: Optional[Path] = None
    iterations: int = 2000
    lr: float = 0.01
    output_path: str = 'fitted_gaussians.pkl'
    num_gpus: int = 1

@dataclass
class ExperimentArgs:
    img_path: Optional[str] = None
    image_folder: Optional[str] = None
    num_points_list: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])
    iterations_list: List[int] = field(default_factory=lambda: [2000, 5000, 10000])
    lr: float = 0.01
    save_imgs: bool = False
    output_log: str = "experiment_results.csv"
    auto_plot: bool = True
    num_gpus: int = 1

@dataclass
class PlotArgs:
    csv_path: str = "experiment_results.csv"

class SimpleTrainer:
    """
    Trains 2D gaussians to fit an RGBA image, using orthographic camera and no culling:
      - camera_model="ortho"
      - eps2d=0.0
      - near_plane=-1e10, far_plane=1e10
      - radius_clip=-1.0
    """

    def __init__(self, gt_image: Tensor, num_points: int = 2000, alpha_mask: Optional[Tensor] = None):
        self.device = gt_image.device
        print(f"Using device: {self.device}")
        self.gt_image = gt_image
        self.alpha_mask = alpha_mask

        self.num_points = num_points
        self.losses = []

        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        # Focal length is mostly irrelevant in orthographic mode, but let's keep it for completeness
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians(alpha_mask=alpha_mask)

    def _init_gaussians(self, alpha_mask: Optional[Tensor] = None):
        """
        Initialize random 2D gaussians *only on the foreground*, if alpha_mask is given.
        alpha_mask: [H,W,1], 0=background, 1=foreground (or 0..1)
        """
        d = 3

        if alpha_mask is None:
            # Random placement in [-1,+1] range
            bd = 2
            self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)
        else:
            # Extract foreground pixels
            mask_2d = alpha_mask[..., 0]  # shape [H,W]
            foreground_indices = torch.nonzero(mask_2d > 0.5, as_tuple=False)  # shape [K,2], (y,x)

            if foreground_indices.shape[0] < self.num_points:
                print("Warning: foreground has fewer pixels than num_points. Some gaussians will overlap.")
            
            # Sample num_points from foreground pixels with replacement
            K = foreground_indices.shape[0]
            chosen = torch.randint(0, K, size=(self.num_points,), device=self.device)
            chosen_xy = foreground_indices[chosen]  # shape [N,2], (y_i, x_i)

            # Convert image pixel coordinates [y, x] to normalized coordinates [-1,+1]
            y = chosen_xy[:, 0].float()
            x = chosen_xy[:, 1].float()

            x_norm = (x / (self.W - 1)) * 2.0 - 1.0
            y_norm = (y / (self.H - 1)) * 2.0 - 1.0

            self.means = torch.stack([x_norm, y_norm], dim=-1)  # shape [N,2]

        # Calculate appropriate scale based on image size and number of Gaussians
        # We want each Gaussian to cover approximately image_area / num_points
        image_area = self.W * self.H
        target_area_per_gaussian = image_area / self.num_points * 1.5
        
        # Convert to normalized coordinates scale
        # For a circle, area = π * r²
        # So r = sqrt(area / π)
        target_radius_pixels = math.sqrt(target_area_per_gaussian / math.pi)
        
        # Convert radius from pixels to normalized coordinates [-1, 1]
        target_radius_norm = target_radius_pixels / min(self.W, self.H) * 2.0
        
        # Set initial scales to this target radius (with small random variation)
        base_scale = target_radius_norm * 0.5  # Make it a bit smaller than theoretical size
        scale_variation = 0.2  # 20% variation
        
        # Initialize scales with the calculated base scale and small random variation
        self.scales = base_scale * (1.0 + scale_variation * (torch.rand(self.num_points, 2, device=self.device) - 0.5))
        
        # Initialize rotations randomly
        self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi
        
        # RGB, opacity (alpha)
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device)

        # Z parameters (fixed)
        self.z_means = torch.zeros(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        # View matrix (fixed)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # Make parameters trainable
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        # Store rasterization metadata
        self.meta = None

        # Visualize the initial Gaussians
        if alpha_mask is not None:
            self.visualize_initial_gaussians()

    def visualize_initial_gaussians(self):
        """Save visualization of initial Gaussians with both points and ellipses."""
        with torch.no_grad():
            # Create output directory
            os.makedirs("comparison", exist_ok=True)
            
            # Convert normalized coordinates to pixel coordinates for visualization
            pixel_means = torch.zeros_like(self.means)
            pixel_means[:, 0] = (self.means[:, 0] + 1) / 2 * (self.W - 1)  # x
            pixel_means[:, 1] = (self.means[:, 1] + 1) / 2 * (self.H - 1)  # y
            
            # Create a scatter plot of initial points
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Show masked GT image as background
            ax.imshow((self.gt_image * self.alpha_mask).detach().cpu().numpy())
                
            # Plot the points
            ax.scatter(
                pixel_means[:, 0].detach().cpu().numpy(),
                pixel_means[:, 1].detach().cpu().numpy(),
                color='red', s=5, alpha=0.7
            )
            ax.set_title("Initial Gaussian Points")
            ax.axis('off')
            
            # Save the initial points visualization
            plt.savefig("comparison/initial_points.png", bbox_inches='tight')
            plt.close(fig)
            print("Saved initial points visualization: comparison/initial_points.png")
            
            # Also visualize Gaussians with ellipses
            covs = self._get_covs()
            self.visualize_gaussians_with_ellipses(
                self.means.detach().cpu().numpy(),
                covs.detach().cpu().numpy(),
                (self.gt_image * self.alpha_mask).detach().cpu().numpy(),
                title="Initial Gaussians",
                save_path="comparison/initial_gaussians.png"
            )
            print("Saved initial Gaussians visualization: comparison/initial_gaussians.png")

    def _get_covs(self):
        """
        Calculate covariance matrices for all Gaussians based on scales and rotations.
        Returns: [N,2,2] tensor of covariance matrices
        """
        N = self.means.shape[0]
        
        # Create rotation matrices
        cos_r = torch.cos(self.rotations)  # [N]
        sin_r = torch.sin(self.rotations)  # [N]
        
        # Stack rotation matrices for all Gaussians
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r,  cos_r], dim=1)
        ], dim=1)  # [N, 2, 2]
        
        # Create scale matrices
        S = torch.diag_embed(self.scales)  # [N, 2, 2]
        
        # Compute covariance matrices: R * S * S^T * R^T
        covs = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)  # [N,2,2]
        
        return covs

    def visualize_masked_results(self, gt_img_masked, out_img_masked, title_prefix=""):
        """
        Create a side-by-side comparison of GT, output, and error map, then save as PNG.
        """
        # Convert PyTorch tensors to numpy if needed
        if hasattr(gt_img_masked, 'detach'):
            gt_img_masked = gt_img_masked.detach().cpu().numpy()
        if hasattr(out_img_masked, 'detach'):
            out_img_masked = out_img_masked.detach().cpu().numpy()

        gt_img_masked = gt_img_masked.clip(0.0, 1.0)
        out_img_masked = out_img_masked.clip(0.0, 1.0)
        diff_map = (gt_img_masked - out_img_masked) ** 2
        mse_value = diff_map.mean()

        # Create comparison image
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(gt_img_masked)
        axs[0].set_title(f"GT Masked")
        axs[1].imshow(out_img_masked)
        axs[1].set_title(f"{title_prefix} Output")
        im_2 = axs[2].imshow(diff_map, cmap='jet', vmin=0, vmax=np.max(diff_map))
        axs[2].set_title(f"MSE={mse_value:.6f}")
        fig.colorbar(im_2, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        os.makedirs("comparison", exist_ok=True)
        save_path = f"comparison/{title_prefix.replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved comparison image: {save_path}")
        plt.close()

    def visualize_gaussians_with_ellipses(self, means, covs, rgb_image=None, title="", save_path=None):
        """
        Draw Gaussian ellipses on an image or blank canvas.
        
        Args:
            means: [N,2] array of Gaussian means
            covs: [N,2,2] array of covariance matrices
            rgb_image: Optional background image
            title: Title for the plot
            save_path: Where to save the visualization
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # If we have a background image, display it
        if rgb_image is not None:
            if hasattr(rgb_image, 'detach'):
                rgb_image = rgb_image.detach().cpu().numpy()
            rgb_image = rgb_image.clip(0, 1)
            ax.imshow(rgb_image)
        else:
            # Otherwise create a blank canvas
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_facecolor('black')
        
        # Draw ellipses for each Gaussian
        for i, (mean, cov) in enumerate(zip(means, covs)):
            # Skip NaN values
            if np.any(np.isnan(mean)) or np.any(np.isnan(cov)):
                continue
            
            # Convert normalized coordinates to pixel coordinates if needed
            if rgb_image is not None and np.max(np.abs(mean)) <= 1.0:
                # Convert from [-1,1] to pixel coordinates
                mean_px = np.array([(mean[0] + 1) / 2 * (self.W - 1), 
                                  (mean[1] + 1) / 2 * (self.H - 1)])
                # Scale covariance matrix to pixel space
                scale_matrix = np.array([[self.W/2, 0], [0, self.H/2]])
                cov_px = scale_matrix @ cov @ scale_matrix.T
            else:
                mean_px = mean
                cov_px = cov
            
            try:
                # Get eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov_px)
                order = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[order]
                eigenvecs = eigenvecs[:, order]
                
                # Skip invalid eigenvalues
                if np.any(eigenvals <= 0):
                    continue
                
                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
                
                # Calculate width and height (2 standard deviations)
                width, height = 2 * 2 * np.sqrt(eigenvals)
                
                # Create ellipse
                ellipse = plt.matplotlib.patches.Ellipse(
                    xy=mean_px, width=width, height=height, angle=angle,
                    edgecolor='red', facecolor='none', linewidth=1, alpha=0.7
                )
                ax.add_patch(ellipse)
            except (np.linalg.LinAlgError, ValueError):
                # Skip problematic Gaussians
                continue
        
        ax.set_title(title)
        if rgb_image is None:
            ax.invert_yaxis()  # Invert y-axis for normalized coordinates
        ax.axis('off')
        
        # Save the figure
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved ellipse visualization to {save_path}")
        
        plt.close(fig)

    def get_gaussians(self) -> Tuple[TwoDGaussians, Optional[TwoDGaussians]]:
        """
        Return (original_gaussians, projected_gaussians).
        If fewer Gaussians are on-screen, subset the attributes for the projected_gaussians.
        """
        with torch.no_grad():
            covs = self._get_covs()

        original_gs = TwoDGaussians(
            means=self.means.detach().cpu().numpy(),
            covs=covs.detach().cpu().numpy(),
            rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy(),
            alpha=torch.sigmoid(self.opacities).detach().cpu().numpy(),
            rotations=self.rotations.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy(),
        )

        projected_gs = None
        if self.meta is not None:
            if 'means2d' in self.meta and 'conics' in self.meta:
                # Extract projected means and conics
                means2d = self.meta['means2d'].detach().cpu().numpy()   # [M,2]
                conics = self.meta['conics'].detach().cpu().numpy()     # [M,3]
                M = means2d.shape[0]

                # Convert conics to covariance matrices
                A = conics[:, 0]
                B = conics[:, 1]
                C = conics[:, 2]
                inv_covs = np.zeros((M, 2, 2))
                inv_covs[:, 0, 0] = A
                inv_covs[:, 0, 1] = B / 2
                inv_covs[:, 1, 0] = B / 2
                inv_covs[:, 1, 1] = C
                covs2d = np.linalg.inv(inv_covs)  # [M,2,2]

                print(f"Projected: {M}/{self.num_points} Gaussians appear on-screen.")

                # Subset RGB, alpha, rotations, scales to match projected Gaussians
                projected_gs = TwoDGaussians(
                    means=means2d,                 # [M,2]
                    covs=covs2d,                   # [M,2,2]
                    rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy()[:M],  # Subset [M,3]
                    alpha=torch.sigmoid(self.opacities).detach().cpu().numpy()[:M],  # Subset [M]
                    rotations=self.rotations.detach().cpu().numpy()[:M],  # Subset [M]
                    scales=self.scales.detach().cpu().numpy()[:M],        # Subset [M,2]
                )

        return original_gs, projected_gs


    def train(
        self,
        max_iterations: int = 40000,
        checkpoint_iterations: List[int] = [5000, 10000, 20000, 30000],
        lr: float = 0.01,
        save_imgs: bool = False,
        output_pkl: Optional[str] = None,
        gt_np: Optional[np.ndarray] = None,
        device: torch.device = torch.device('cpu'),
    ) -> List[dict]:
        """
        Train loop with RGBA rendering (pre-multiplied alpha).
        - We pass 4 channels (r,g,b,a) to rasterization,
            and retrieve out_rgba = [H,W,4].
        - Then final color = out_rgb * out_alpha, which is compared
            to gt_rgb * alpha_mask if present.
        """
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()
        frames = []
        times = [0.0, 0.0]

        # Camera intrinsics
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)

        final_out_img = None
        results = []

        for i in range(max_iterations):
            start = time.time()
            optimizer.zero_grad()

            # --- 3D transforms ---
            means_3d = torch.cat([self.means, self.z_means], dim=1)   # [N,3]
            scales_3d= torch.cat([self.scales, self.scales_z], dim=1)
            quats = torch.stack([
                torch.cos(self.rotations / 2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations / 2)
            ], dim=1)
            quats_norm = quats / quats.norm(dim=-1, keepdim=True)

            # --- RGBA 4ch: (r,g,b,a) in [0..1]
            rgba_for_raster = torch.cat([
                torch.sigmoid(self.rgbs),                      # shape [N,3]
                torch.sigmoid(self.opacities).unsqueeze(-1),   # shape [N,1]
            ], dim=-1)  # => shape [N,4]

            # Use dummy opacities (all 1s) since we're using RGBA with alpha
            dummy_opacities = torch.ones_like(self.opacities)

            # --- Rasterization ---
            # Expecting out_img shape [H,W,4] with RGBA channels
            renders, _, meta = rasterization(
                means_3d,
                quats_norm,
                scales_3d,
                dummy_opacities,
                rgba_for_raster,      # [N,4]
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                camera_model="ortho"  # Use orthographic camera model
            )

            out_rgba = renders[0]  # shape [H,W,4]
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[0] += time.time() - start

            # --- Extract RGB and alpha components ---
            out_rgb   = out_rgba[..., :3]   # [H,W,3]
            out_alpha = out_rgba[..., 3:4]  # [H,W,1]
            
            # --- Apply pre-multiplied alpha (background becomes black) ---
            out_pre = out_rgb * out_alpha  # [H,W,3], pre-multiplied

            # --- GT side also uses pre-multiplied alpha if mask exists ---
            if self.alpha_mask is not None:
                # shape [H,W,3]
                gt_pre = self.gt_image * self.alpha_mask
            else:
                gt_pre = self.gt_image

            # --- Calculate MSE loss ---
            loss = mse_loss_fn(out_pre, gt_pre)

            start = time.time()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[1] += time.time() - start

            self.losses.append(loss.item())
            optimizer.step()
            self.meta = meta

            # --- Print progress ---
            if i % 100 == 0:
                print(f"Iter {i}/{max_iterations}, Loss: {loss.item():.6f}")
                
                # Save intermediate visualizations
                if save_imgs and i % 1000 == 0:
                    with torch.no_grad():
                        # Compare GT and rendered output
                        self.visualize_masked_results(
                            gt_pre,
                            out_pre,
                            title_prefix=f"Iter_{i}"
                        )
                        
                        # Visualize Gaussians if we have projected info
                        if 'means2d' in meta:
                            means2d = meta['means2d'].detach().cpu().numpy()
                            covs = self._get_covs().detach().cpu().numpy()
                            
                            self.visualize_gaussians_with_ellipses(
                                means2d,
                                covs,
                                out_pre.detach().cpu().numpy(),
                                title=f"Gaussians at Iteration {i}",
                                save_path=f"comparison/gaussians_iter_{i}.png"
                            )

            # --- Checkpoint & metrics ---
            if (i + 1) in checkpoint_iterations:
                # Extract output image for metrics
                final_out_img = out_pre.detach().cpu().numpy()  # [H,W,3]

                if gt_np is not None:
                    if self.alpha_mask is not None:
                        # Apply alpha to GT for metrics
                        alpha_np = self.alpha_mask.detach().cpu().numpy()
                        gt_pre_np = gt_np * alpha_np
                    else:
                        gt_pre_np = gt_np

                    # Calculate metrics
                    psnr_val = calculate_psnr(final_out_img, gt_pre_np, max_val=1.0)
                    lpips_val = calculate_lpips(final_out_img, gt_pre_np, device=device)
                    ssim_val = calculate_ssim(final_out_img, gt_pre_np, device=device)
                else:
                    psnr_val = -1
                    lpips_val = -1
                    ssim_val = -1

                results.append({
                    "iteration": i + 1,
                    "psnr": psnr_val,
                    "lpips": lpips_val,
                    "ssim": ssim_val,
                })
                print(f"Iter {i+1}/{max_iterations}, Loss={loss.item():.6f}, PSNR={psnr_val:.4f}, LPIPS={lpips_val:.4f}, SSIM={ssim_val:.4f}")

            # --- Save frames for GIF ---
            if save_imgs and ((i + 1) % 5 == 0):
                frame_rgb = (out_pre.detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                frames.append(frame_rgb)

        print(f"[Timing] Total: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
        print(f"[Timing] Per step: R={times[0]/max_iterations:.6f}, B={times[1]/max_iterations:.6f}")

        # --- Final visualization ---
        if save_imgs:
            self.visualize_masked_results(
                gt_pre.detach().cpu().numpy(),
                out_pre.detach().cpu().numpy(),
                title_prefix="Final"
            )

        # --- Save final Gaussians ---
        original_gs, projected_gs = self.get_gaussians()
        if output_pkl is not None:
            self.save_gaussians(original_gs, projected_gs, output_pkl)

        # --- Optionally save GIF ---
        if save_imgs and frames:
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            from PIL import Image
            frames_pil = [Image.fromarray(f) for f in frames]
            frames_pil[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames_pil[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        return results


    def plot_loss(self, save_path='loss_curve.png'):
        plt.figure(figsize=(8,5))
        plt.plot(self.losses, label="MSE Loss")
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved: {save_path}")

    def save_gaussians(self, original_gs: TwoDGaussians, projected_gs: Optional[TwoDGaussians], path: str):
        import pickle
        # Include both the original and projected Gaussians
        data = {
            'original_gaussians': original_gs,
            'projected_gaussians': projected_gs,
            'viewmat': self.viewmat.detach().cpu().numpy(),
            'K': torch.tensor([
                [self.focal, 0, self.W/2],
                [0, self.focal, self.H/2],
                [0, 0, 1],
            ]).detach().cpu().numpy(),
            'image_size': (self.W, self.H)  # Explicitly save the image dimensions
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Gaussians to: {path} with image size {(self.W, self.H)}")


def image_path_to_tensor(image_path: Path, device: torch.device = None) -> Tensor:
    """Loads an image and returns shape [H,W,3] in [0..1]."""
    import torchvision.transforms as transforms
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(img).permute(1,2,0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def rgba_path_to_tensor(image_path: Path, device: torch.device=None) -> Tuple[Tensor, Tensor]:
    """
    Load a 4-channel RGBA image and split into (rgb, alpha).
      - rgb:   [H,W,3] in [0..1]
      - alpha: [H,W,1] in [0..1], 0=background, 1=foreground
    """
    from torchvision import transforms
    img = Image.open(image_path).convert("RGBA")  # 4ch
    t = transforms.ToTensor()(img)  # shape [4,H,W], in [0..1]
    t = t.permute(1,2,0)           # [H,W,4]
    rgb   = t[..., :3]            # [H,W,3]
    alpha = t[...,  3:]           # [H,W,1]

    if device is not None:
        rgb   = rgb.to(device)
        alpha = alpha.to(device)
    return rgb, alpha


def plot_results_from_csv(csv_path: str) -> None:
    """Reads CSV experiment results and plots PSNR, LPIPS, SSIM vs. num_points for each iteration count."""
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "image_name": row.get("image_name","unknown"),
                "num_points": int(row["num_points"]),
                "iterations": int(row["iterations"]),
                "psnr": float(row["psnr"]),
                "lpips": float(row["lpips"]),
                "ssim": float(row["ssim"])
            })

    from collections import defaultdict
    results_by_iterations = defaultdict(list)
    for entry in data:
        results_by_iterations[entry["iterations"]].append(entry)

    # Sort each group by num_points
    for iters, group in results_by_iterations.items():
        group.sort(key=lambda x: x["num_points"])

    # Plot PSNR
    plt.figure(figsize=(10,6))
    for iters, group in results_by_iterations.items():
        x = [g["num_points"] for g in group]
        y = [g["psnr"] for g in group]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.xlabel("num_points")
    plt.ylabel("PSNR")
    plt.title("PSNR vs num_points")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_num_points.png")
    plt.close()
    print("PSNR plot saved: psnr_vs_num_points.png")

    # Plot LPIPS
    plt.figure(figsize=(10,6))
    for iters, group in results_by_iterations.items():
        x = [g["num_points"] for g in group]
        y = [g["lpips"] for g in group]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.xlabel("num_points")
    plt.ylabel("LPIPS (Lower is better)")
    plt.title("LPIPS vs num_points")
    plt.grid(True)
    plt.legend()
    plt.savefig("lpips_vs_num_points.png")
    plt.close()
    print("LPIPS plot saved: lpips_vs_num_points.png")

    # Plot SSIM
    plt.figure(figsize=(10,6))
    for iters, group in results_by_iterations.items():
        x = [g["num_points"] for g in group]
        y = [g["ssim"] for g in group]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.xlabel("num_points")
    plt.ylabel("SSIM (Higher is better)")
    plt.title("SSIM vs num_points")
    plt.grid(True)
    plt.legend()
    plt.savefig("ssim_vs_num_points.png")
    plt.close()
    print("SSIM plot saved: ssim_vs_num_points.png")

def load_images_from_folder(folder: Path) -> List[Path]:
    """Return sorted list of valid image paths in the folder."""
    valid_exts = ('.jpg','.jpeg','.png','.bmp','.tiff')
    image_paths = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts:
            image_paths.append(f)
    image_paths.sort()
    return image_paths

def generate_synthetic_image(device: torch.device, height=256, width=256) -> Tensor:
    """Generate a 256x256 synthetic image: half is red, half is blue, rest is white."""
    img = torch.ones((height, width,3), device=device)
    # top-left quadrant = red
    img[:height//2, :width//2,:] = torch.tensor([1.0,0.0,0.0], device=device)
    # bottom-right quadrant = blue
    img[height//2:, width//2:,:] = torch.tensor([0.0,0.0,1.0], device=device)
    return img

def _train_on_image(
    img_path: Path,
    num_points: int,
    iterations: int,
    lr: float,
    save_imgs: bool,
    output_pkl: Optional[str],
    gpu_id: int
) -> str:
    """
    Worker function for 'train' subcommand on a single image using a specific GPU.
    Returns the pkl filename that was saved (if any).
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu_id}] Training on {img_path.name} ...")

    # Load RGBA image
    gt_rgb, alpha_mask = rgba_path_to_tensor(img_path, device=device)
    gt_np = gt_rgb.cpu().numpy()

    # Create trainer with alpha mask
    trainer = SimpleTrainer(
        gt_image=gt_rgb,
        num_points=num_points,
        alpha_mask=alpha_mask
    )

    # Train with checkpoints at the requested iterations
    trainer.train(
        max_iterations=iterations,
        checkpoint_iterations=[iterations],
        lr=lr,
        save_imgs=save_imgs,
        output_pkl=output_pkl,
        gt_np=gt_np,
        device=device
    )
    trainer.plot_loss(save_path=str(Path(output_pkl).with_suffix('.png')))
    return output_pkl

def _experiment_on_image(
    img_path: Path,
    num_points_list: List[int],
    iterations_list: List[int],
    lr: float,
    save_imgs: bool,
    gpu_id: int
) -> List[dict]:
    """
    Train once for each num_points up to max(iterations_list). For each trainer.run(...),
    we capture the checkpoint metrics that includes iteration, psnr, lpips, ssim.
    Then we store them in results_for_image with the needed fields for CSV.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu_id}] Experiment on {img_path.name} ...")

    # Load RGBA image
    gt_rgb, alpha_mask = rgba_path_to_tensor(img_path, device=device)
    gt_np = gt_rgb.cpu().numpy()

    results_for_image = []

    # For each npoints, run a single training up to max(iterations_list)
    for npoints in num_points_list:
        print(f"[GPU {gpu_id}] -> {img_path.name}, npoints={npoints}")
        trainer = SimpleTrainer(
            gt_image=gt_rgb, 
            num_points=npoints,
            alpha_mask=alpha_mask
        )

        # Train until the maximum iteration with checkpoints at each specified iteration
        all_metrics = trainer.train(
            max_iterations=max(iterations_list),
            checkpoint_iterations=iterations_list,
            lr=lr,
            save_imgs=save_imgs,
            gt_np=gt_np,
            device=device,
        )

        # Gather metrics from checkpoints
        for metrics in all_metrics:
            results_for_image.append({
                "image_name": img_path.name,
                "num_points": npoints,
                "iterations": metrics["iteration"],
                "psnr": metrics["psnr"],
                "lpips": metrics["lpips"],
                "ssim": metrics["ssim"],
            })

    return results_for_image




def main():
    parser = argparse.ArgumentParser(description="Fitting 2D Gaussians to an Image (single or multiple images)")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # --- Train subcommand ---
    parser_train = subparsers.add_parser('train', help='Run a single training session (or multiple if folder).')
    parser_train.add_argument('--height', type=int, default=256, help='Height if synthetic image is used')
    parser_train.add_argument('--width', type=int, default=256, help='Width if synthetic image is used')
    parser_train.add_argument('--num_points', type=int, default=200, help='Number of Gaussian points')
    parser_train.add_argument('--save_imgs', action='store_true', help='Save rendered images as GIF')
    parser_train.add_argument('--img_path', type=Path, default=None, help='Path to a single ground truth image')
    parser_train.add_argument('--image_folder', type=Path, default=None, help='Folder containing multiple images')
    parser_train.add_argument('--iterations', type=int, default=2000, help='Number of training iterations')
    parser_train.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser_train.add_argument('--output_path', type=str, default='fitted_gaussians.pkl', help='Path to save PKL')
    parser_train.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')

    # --- Experiment subcommand ---
    parser_experiment = subparsers.add_parser('experiment', help='Run multiple training sessions w/ different parameters.')
    parser_experiment.add_argument('--img_path', type=str, default=None, help='Path to a single image')
    parser_experiment.add_argument('--image_folder', type=str, default=None, help='Folder of images to run experiment on')
    parser_experiment.add_argument('--num_points_list', type=int, nargs='+', default=[100, 200, 500, 1000, 2000],
                                   help='List of number of Gaussian points to experiment with')
    parser_experiment.add_argument('--iterations_list', type=int, nargs='+', default=[2000, 5000, 10000],
                                   help='List of iteration counts to experiment with')
    parser_experiment.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser_experiment.add_argument('--save_imgs', action='store_true', help='Save rendered images as GIF')
    parser_experiment.add_argument('--output_log', type=str, default='experiment_results.csv', help='Path to save CSV')
    parser_experiment.add_argument('--auto_plot', action='store_true', help='Generate plots after experiments')
    parser_experiment.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')

    # --- Plot subcommand ---
    parser_plot = subparsers.add_parser('plot', help='Plot results from a CSV file')
    parser_plot.add_argument('--csv_path', type=str, default='experiment_results.csv', help='Path to CSV')

    args = parser.parse_args()

    # Decide which subcommand was called
    if args.command == 'train':
        if args.image_folder is not None:
            # Train on every image in folder, optionally using multiple GPUs
            image_paths = load_images_from_folder(args.image_folder)
            if len(image_paths) == 0:
                print(f"No valid images found in {args.image_folder}")
                sys.exit(1)

            output_folder = Path.cwd() / "train_results"
            output_folder.mkdir(parents=True, exist_ok=True)

            # If multiple GPUs, distribute images among them
            if args.num_gpus > 1:
                futures = []
                with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
                    for i, img_path in enumerate(image_paths):
                        gpu_id = i % args.num_gpus
                        out_name = output_folder / f"{img_path.stem}_fitted_gaussians.pkl"
                        futures.append(
                            executor.submit(
                                _train_on_image,
                                img_path=img_path,
                                num_points=args.num_points,
                                iterations=args.iterations,
                                lr=args.lr,
                                save_imgs=args.save_imgs,
                                output_pkl=str(out_name),
                                gpu_id=gpu_id
                            )
                        )
                    for f in as_completed(futures):
                        print(f"Done: {f.result()}")
            else:
                # Single GPU
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                for i, img_path in enumerate(image_paths):
                    print(f"[TRAIN single-GPU] Processing {img_path.name}")
                    out_name = output_folder / f"{img_path.stem}_fitted_gaussians.pkl"
                    _train_on_image(
                        img_path=img_path,
                        num_points=args.num_points,
                        iterations=args.iterations,
                        lr=args.lr,
                        save_imgs=args.save_imgs,
                        output_pkl=str(out_name),
                        gpu_id=0
                    )
        else:
            # Single image
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.img_path:
                # Load RGBA image
                gt_rgb, alpha_mask = rgba_path_to_tensor(args.img_path, device=device)
            else:
                # Generate synthetic image with no alpha mask
                gt_rgb = generate_synthetic_image(device, height=args.height, width=args.width)
                alpha_mask = None

            trainer = SimpleTrainer(
                gt_image=gt_rgb, 
                num_points=args.num_points,
                alpha_mask=alpha_mask
            )
            
            trainer.train(
                max_iterations=args.iterations,
                checkpoint_iterations=[args.iterations],
                lr=args.lr,
                save_imgs=args.save_imgs,
                output_pkl=args.output_path,
                device=device
            )
            trainer.plot_loss(save_path="loss_curve.png")

    elif args.command == 'experiment':
        # Define the fieldnames for CSV
        fieldnames = ["image_name", "num_points", "iterations", "psnr", "lpips", "ssim"]

        # Initialize aggregated dictionary
        aggregated = defaultdict(list)

        if args.image_folder is not None:
            folder = Path(args.image_folder)
            image_paths = load_images_from_folder(folder)
            if len(image_paths) == 0:
                print(f"No valid images found in {folder}")
                sys.exit(1)

            with open(args.output_log, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            futures = []
            if args.num_gpus > 1:
                with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
                    for i, img_path in enumerate(image_paths):
                        gpu_id = i % args.num_gpus
                        fut = executor.submit(
                            _experiment_on_image,
                            img_path=img_path,
                            num_points_list=args.num_points_list,
                            iterations_list=args.iterations_list,
                            lr=args.lr,
                            save_imgs=args.save_imgs,
                            gpu_id=gpu_id
                        )
                        futures.append(fut)
                    for fut in as_completed(futures):
                        results_for_image = fut.result()
                        with open(args.output_log, "a", newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            for result in results_for_image:
                                try:
                                    # Convert metrics to float
                                    psnr_val = float(result["psnr"])
                                    lpips_val = float(result["lpips"])
                                    ssim_val = float(result["ssim"])

                                    # Write row to CSV
                                    writer.writerow({
                                        "image_name": result["image_name"],
                                        "num_points": result["num_points"],
                                        "iterations": result["iterations"],
                                        "psnr": psnr_val,
                                        "lpips": lpips_val,
                                        "ssim": ssim_val
                                    })

                                    # Append to aggregated for averaging
                                    aggregated[(result["num_points"], result["iterations"])].append(
                                        (psnr_val, lpips_val, ssim_val)
                                    )
                                except ValueError as e:
                                    print(f"Error processing result {result}: {e}")
                                    continue  # Skip invalid entries
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                for img_path in image_paths:
                    results_for_image = _experiment_on_image(
                        img_path=img_path,
                        num_points_list=args.num_points_list,
                        iterations_list=args.iterations_list,
                        lr=args.lr,
                        save_imgs=args.save_imgs,
                        gpu_id=0
                    )
                    with open(args.output_log, "a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for result in results_for_image:
                            try:
                                # Convert metrics to float
                                psnr_val = float(result["psnr"])
                                lpips_val = float(result["lpips"])
                                ssim_val = float(result["ssim"])

                                # Write row to CSV
                                writer.writerow({
                                    "image_name": result["image_name"],
                                    "num_points": result["num_points"],
                                    "iterations": result["iterations"],
                                    "psnr": psnr_val,
                                    "lpips": lpips_val,
                                    "ssim": ssim_val
                                })

                                # Append to aggregated for averaging
                                aggregated[(result["num_points"], result["iterations"])].append(
                                    (psnr_val, lpips_val, ssim_val)
                                )
                            except ValueError as e:
                                print(f"Error processing result {result}: {e}")
                                continue  # Skip invalid entries

            # Compute average results
            avg_out = Path(args.output_log).parent / "experiment_results_averaged.csv"
            with open(avg_out, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(["num_points", "iterations", "psnr_mean", "lpips_mean", "ssim_mean", "count"])
                for (npts, iters), vals in aggregated.items():
                    if not vals:
                        continue  # Skip if no valid data
                    vals_array = np.array(vals)
                    means = vals_array.mean(axis=0)
                    w.writerow([npts, iters, means[0], means[1], means[2], len(vals)])

            print(f"Averaged results saved to {avg_out}")

            if args.auto_plot:
                plot_results_from_csv(args.output_log)
        else:
            # Single image experiment
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.img_path:
                img_path = Path(args.img_path)
                gt_rgb, alpha_mask = rgba_path_to_tensor(img_path, device=device)
                image_name = img_path.name
            else:
                gt_rgb = generate_synthetic_image(device)
                alpha_mask = None
                image_name = "synthetic"

            with open(args.output_log, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            for np_ in args.num_points_list:
                for it_ in args.iterations_list:
                    print(f"[EXPERIMENT single] npoints={np_}, iters={it_}")
                    trainer = SimpleTrainer(
                        gt_image=gt_rgb, 
                        num_points=np_,
                        alpha_mask=alpha_mask
                    )
                    
                    # Train to the specified iteration with a single checkpoint
                    metrics = trainer.train(
                        max_iterations=it_,
                        checkpoint_iterations=[it_],
                        lr=args.lr,
                        save_imgs=args.save_imgs,
                        device=device
                    )
                    
                    # Get metrics from the checkpoint
                    if metrics:
                        metric = metrics[0]  # Just one checkpoint
                        with open(args.output_log, "a", newline='') as f:
                            w = csv.DictWriter(f, fieldnames=fieldnames)
                            w.writerow({
                                "image_name": image_name,
                                "num_points": np_,
                                "iterations": it_,
                                "psnr": metric["psnr"],
                                "lpips": metric["lpips"],
                                "ssim": metric["ssim"]
                            })

            print(f"Results saved to {args.output_log}")
            if args.auto_plot:
                plot_results_from_csv(args.output_log)

    elif args.command == 'plot':
        plot_results_from_csv(args.csv_path)
    else:
        print("Unknown command. Use 'train', 'experiment', or 'plot'.")
        sys.exit(1)

if __name__ == "__main__":
    main()