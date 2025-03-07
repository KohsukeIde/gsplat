#!/usr/bin/env python

import math
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor, optim

from gsplat import rasterization
from twodgs import TwoDGaussians


def rgba_path_to_tensor(image_path: Path, device: torch.device = None):
    """
    4チャンネル RGBA 画像を読み込み、(rgb, alpha) の2つに分けて返す。
      - rgb:   [H,W,3] in [0..1]
      - alpha: [H,W,1] in [0..1], 0=背景, 1=前景など
    """
    from torchvision import transforms
    img = Image.open(image_path).convert("RGBA")  # RGBA
    t = transforms.ToTensor()(img)  # shape [4,H,W], in [0..1]
    # 転置して [H,W,4]
    t = t.permute(1, 2, 0)
    rgb = t[..., :3]  # [H,W,3]
    alpha = t[..., 3:]  # [H,W,1]

    if device is not None:
        rgb = rgb.to(device)
        alpha = alpha.to(device)

    return rgb, alpha


class SimpleTrainer:
    """
    Trains 2D gaussians to fit an RGBA image, 
    and uses (color * alpha) as the final output for MSE loss.
    """

    def __init__(
        self,
        gt_rgb: Tensor,       # [H,W,3]
        alpha_mask: Tensor,   # [H,W,1], 0=背景,1=前景
        num_points: int = 2000,
    ):
        self.device = gt_rgb.device
        print(f"{self.device=}")

        # (1) GT画像を"pre-multiplied alpha"として扱う => 背景ピクセルは(0,0,0)
        self.gt_image = gt_rgb
        self.alpha_mask = alpha_mask  # 後でサンプリングや可視化にも使う
        self.num_points = num_points
        self.losses = []

        self.H, self.W = gt_rgb.shape[0], gt_rgb.shape[1]
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        # ガウス初期化
        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize random 2D gaussians in the *foreground* region (alpha=1)."""
        d = 3

        if self.alpha_mask is not None:
            # 前景画素を抽出
            mask_2d = self.alpha_mask[..., 0]  # shape [H,W]
            foreground_indices = torch.nonzero(mask_2d > 0.5, as_tuple=False)  # [K,2], (y,x)

            K = foreground_indices.shape[0]
            if K < self.num_points:
                print("Warning: foreground has fewer pixels than num_points. Overlaps will occur.")

            chosen = torch.randint(0, K, size=(self.num_points,), device=self.device)
            chosen_xy = foreground_indices[chosen]  # [N,2], (y_i, x_i)

            # 画像座標 [y,x] → 正規化座標 [-1,+1]
            y = chosen_xy[:, 0].float()
            x = chosen_xy[:, 1].float()
            x_norm = (x / (self.W - 1)) * 2.0 - 1.0
            y_norm = (y / (self.H - 1)) * 2.0 - 1.0
            self.means = torch.stack([x_norm, y_norm], dim=-1)  # [N,2]
        else:
            # 従来通り [-1,+1] にランダム
            bd = 2
            self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)

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
        
        # RGB, α
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device)

        # Zパラメータ (固定 or 自由)
        self.z_means = torch.zeros(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        # ビュー行列 (固定)
        self.viewmat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device)

        # 学習対象に設定
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True

        self.viewmat.requires_grad = False
        self.meta = None
        
        # Save initial Gaussian visualization
        with torch.no_grad():
            # Save visualization of initial points (before any rendering)
            os.makedirs("comparison", exist_ok=True)
            
            # Convert normalized coordinates to pixel coordinates for visualization
            pixel_means = torch.zeros_like(self.means)
            pixel_means[:, 0] = (self.means[:, 0] + 1) / 2 * (self.W - 1)  # x
            pixel_means[:, 1] = (self.means[:, 1] + 1) / 2 * (self.H - 1)  # y
            
            # Create a simple scatter plot of initial points
            fig, ax = plt.subplots(figsize=(10, 10))
            if self.alpha_mask is not None:
                # Show the masked GT image as background
                ax.imshow((self.gt_image * self.alpha_mask).detach().cpu().numpy())
            else:
                ax.set_facecolor('black')
                
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
            
            # Also save initial Gaussians with ellipses
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
        covs = torch.zeros(N, 2, 2, device=self.device)
        
        # Convert scales to standard deviations
        stds = self.scales  # [N,2]
        
        # Create rotation matrices
        cos_r = torch.cos(self.rotations)  # [N]
        sin_r = torch.sin(self.rotations)  # [N]
        
        # Fill in covariance matrices
        for i in range(N):
            # Create rotation matrix
            R = torch.tensor([
                [cos_r[i], -sin_r[i]],
                [sin_r[i], cos_r[i]]
            ], device=self.device)
            
            # Create diagonal matrix with squared standard deviations
            S = torch.diag(stds[i]**2)
            
            # Compute covariance matrix: R * S * R^T
            covs[i] = R @ S @ R.t()
        
        return covs

    def get_gaussians(self) -> tuple:
        """Return (original_gaussians, projected_gaussians)."""
        with torch.no_grad():
            covs = self._get_covs()

        original_gs = TwoDGaussians(
            means=self.means.detach().cpu().numpy(),
            covs=covs.detach().cpu().numpy(),
            rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy(),
            alpha=torch.sigmoid(self.opacities).detach().cpu().numpy(),
            rotations=self.rotations.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy()
        )

        projected_gs = None
        if self.meta is not None:
            if 'means2d' in self.meta and 'conics' in self.meta:
                means2d = self.meta['means2d'].detach().cpu().numpy()
                conics  = self.meta['conics'].detach().cpu().numpy()
                M = means2d.shape[0]

                A = conics[:, 0]
                B = conics[:, 1]
                C = conics[:, 2]
                inv_covs = np.zeros((M,2,2))
                inv_covs[:,0,0] = A
                inv_covs[:,0,1] = B/2
                inv_covs[:,1,0] = B/2
                inv_covs[:,1,1] = C
                covs2d = np.linalg.inv(inv_covs)  # [M,2,2]

                rgbs = torch.sigmoid(self.rgbs).detach().cpu().numpy()[:M]
                alphas = torch.sigmoid(self.opacities).detach().cpu().numpy()[:M]
                projected_gs = TwoDGaussians(
                    means=means2d,
                    covs=covs2d,
                    rgb=rgbs,
                    alpha=alphas,
                    rotations=self.rotations.detach().cpu().numpy()[:M],
                    scales=self.scales.detach().cpu().numpy()[:M],
                )

        return original_gs, projected_gs


    def visualize_masked_results(self, gt_img_masked, out_img_masked, title_prefix=""):
        """
        2枚の画像と誤差マップを並べて比較をPNG保存するユーティリティ。
        """
        # もしPyTorch Tensorなら detach + cpu + numpy に変換
        if hasattr(gt_img_masked, 'detach'):
            gt_img_masked = gt_img_masked.detach().cpu().numpy()
        if hasattr(out_img_masked, 'detach'):
            out_img_masked = out_img_masked.detach().cpu().numpy()

        gt_img_masked   = gt_img_masked.clip(0.0, 1.0)
        out_img_masked  = out_img_masked.clip(0.0, 1.0)
        diff_map        = (gt_img_masked - out_img_masked) ** 2
        mse_value       = diff_map.mean()

        # Ensure all images have the same dimensions for consistent display
        H, W = gt_img_masked.shape[:2]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(gt_img_masked)
        axs[0].set_title(f"GT Masked")
        axs[1].imshow(out_img_masked)
        axs[1].set_title(f"{title_prefix} Out")
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

    def save_gaussians(self, orig_gs: TwoDGaussians, proj_gs: TwoDGaussians, path: str):
        import pickle
        data = {
            "original_gaussians": orig_gs,
            "projected_gaussians": proj_gs,
            "viewmat": self.viewmat.detach().cpu().numpy(),
            "K": torch.tensor([
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ], device=self.device).detach().cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved gaussians to {path}")


    def train(
        self,
        iterations: int = 10000,
        lr: float = 0.01,
        save_imgs: bool = False,
    ):
        """
        フィッティングした RGBA から最終的に (RGB * α) を取り出し，
        GT側 (gt_rgb * alpha_mask) と比較する。
        """
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()

        frames = []
        times = [0,0]

        # Camera matrix for orthographic projection
        K = torch.tensor([
            [self.focal, 0, self.W/2],
            [0, self.focal, self.H/2],
            [0, 0, 1],
        ], device=self.device)

        # Store loss values for plotting
        self.losses = []

        for it in range(iterations):
            start = time.time()
            optimizer.zero_grad()

            # 3Dパラメータ
            means_3d = torch.cat([self.means, self.z_means], dim=-1)  # [N,3]
            scales_3d= torch.cat([self.scales, self.scales_z], dim=-1)
            quats = torch.stack([
                torch.cos(self.rotations/2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations/2)
            ], dim=1)
            quats_norm = quats / quats.norm(dim=-1, keepdim=True)

            # RGBA (N,4) : (r,g,b, alpha)
            rgba_for_raster = torch.cat([
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities).unsqueeze(-1)
            ], dim=-1)  # => shape [N,4], each in [0..1]

            # dummy_opacities は rasterization の opacities引数に相当
            #   (実際には使われず RGBAのAを使う設定とする)
            dummy_opacities = torch.ones_like(self.opacities)

            # "RGB" モードであっても 4ch を渡せば out[..., :3], out[...,3] が使える 
            renders, alpha_out, meta = rasterization(
                means_3d,
                quats_norm,
                scales_3d,
                dummy_opacities,
                rgba_for_raster,   # [N,4]
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                render_mode="RGB",  # 出力 shape: [H,W,4] (RGBA)
                camera_model="ortho"  # Use orthographic camera model
            )

            out_rgba  = renders[0]               # [H,W,4]
            out_rgb   = out_rgba[..., :3]        # [H,W,3]
            out_alpha = out_rgba[...,  3:4]      # [H,W,1]
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # (1) 「フィッティングした色 × フィッティングした透明度」: pre-multiplied alpha
            out_pre_mult = out_rgb * out_alpha   # [H,W,3]

            # (2) 「GT色 × GTマスク」と比較
            loss = mse_loss_fn(out_pre_mult, self.gt_image * self.alpha_mask)
            self.losses.append(loss.item())

            # 勾配計算
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            # 最適化ステップ
            optimizer.step()

            # 途中経過表示
            if it % 100 == 0:
                print(f"Iter {it}, Loss: {loss.item():.6f}")
                
                # Save visualization at specific iterations
                if save_imgs and it % 1000 == 0:
                    with torch.no_grad():
                        # Visualize comparison between GT and rendered result
                        self.visualize_masked_results(
                            self.gt_image * self.alpha_mask,
                            out_pre_mult,
                            title_prefix=f"Iter_{it}"
                        )
                        
                        # Check if meta contains projected information for visualization
                        if 'means2d' in meta:
                            # Safely extract means2d
                            means2d = meta['means2d'].detach().cpu().numpy()
                            
                            # For visualization without conics, use original covariance matrices
                            covs = self._get_covs().detach().cpu().numpy()
                            
                            # Visualize with ellipses
                            self.visualize_gaussians_with_ellipses(
                                means2d,
                                covs,
                                out_pre_mult.detach().cpu().numpy(),
                                title=f"Gaussians at Iteration {it}",
                                save_path=f"comparison/gaussians_iter_{it}.png"
                            )
                            print(f"Saved Gaussian visualization at iteration {it}")

            # Save visualization after the first iteration
            if it == 0 and save_imgs:
                # Store meta information for visualization
                self.meta = meta
                
                # Visualize comparison between GT and rendered result
                self.visualize_masked_results(
                    self.gt_image * self.alpha_mask,
                    out_pre_mult,
                    title_prefix="Iter_0"
                )
                
                # For first iteration, use original covariance matrices
                covs = self._get_covs().detach().cpu().numpy()
                
                # If meta contains projected means, use those
                if 'means2d' in meta:
                    means2d = meta['means2d'].detach().cpu().numpy()
                    
                    # Visualize with ellipses
                    self.visualize_gaussians_with_ellipses(
                        means2d,
                        covs,
                        out_pre_mult.detach().cpu().numpy(),
                        title="Gaussians at Iteration 0",
                        save_path="comparison/gaussians_iter_0.png"
                    )
                    print("Saved initial Gaussian visualization")

        # GIF 保存
        if save_imgs and len(frames) > 0:
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        # 最終結果を可視化
        if save_imgs:
            self.visualize_masked_results(
                self.gt_image * self.alpha_mask,
                out_pre_mult,
                title_prefix="Final"
            )

        print(f"Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
        return self.get_gaussians()


    def plot_loss(self, save_path="loss_curve.png"):
        plt.figure(figsize=(8,5))
        plt.plot(self.losses, label="MSE Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved: {save_path}")

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
        # Create figure and axis with consistent size
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # If we have a background image, display it
        if rgb_image is not None:
            if hasattr(rgb_image, 'detach'):
                rgb_image = rgb_image.detach().cpu().numpy()
            rgb_image = rgb_image.clip(0, 1)
            ax.imshow(rgb_image)
        else:
            # Otherwise create a blank canvas with the right dimensions
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
            except (np.linalg.LinAlgError, ValueError) as e:
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


def main():
    parser = argparse.ArgumentParser(description="Train 2D Gaussians with pre-multiplied alpha loss.")
    parser.add_argument("--img_path", type=str, required=True, help="Path to RGBA image")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save_imgs", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) RGBA読み込み
    img_path_obj = Path(args.img_path)
    rgb, alpha = rgba_path_to_tensor(img_path_obj, device=device)

    # 2) Trainer作成
    trainer = SimpleTrainer(gt_rgb=rgb, alpha_mask=alpha, num_points=args.num_points)

    # 3) 学習
    orig_gs, proj_gs = trainer.train(
        iterations=args.iterations,
        lr=args.lr,
        save_imgs=args.save_imgs,
    )

    # 4) Loss plot
    trainer.plot_loss("loss_curve.png")


if __name__ == "__main__":
    main()
