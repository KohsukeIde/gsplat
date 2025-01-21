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
    Trains 2D gaussians to fit an image, using orthographic camera and no culling:
      - camera_model="ortho"
      - eps2d=0.0
      - near_plane=-1e10, far_plane=1e10
      - radius_clip=-1.0
    """

    def __init__(self, gt_image: Tensor, num_points: int = 2000):
        self.device = gt_image.device
        print(f"Using device: {self.device}")
        self.gt_image = gt_image
        self.num_points = num_points
        self.losses = []

        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        # Focal length is mostly irrelevant in orthographic mode, but let's keep it for completeness
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize random 2D gaussians, put them in 3D at z=+1."""
        bd = 2
        d = 3

        # means in [-1, +1] for (x,y)
        self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)  # [N,2]
        self.scales = torch.rand(self.num_points, 2, device=self.device)  # [N,2]
        self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi  # [N]
        self.rgbs = torch.rand(self.num_points, d, device=self.device)  # [N,3]
        self.opacities = torch.ones(self.num_points, device=self.device)  # [N]

        # Z=+1 so they're clearly in front for orthographic
        self.z_means = torch.ones(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        # Orthographic camera
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )  # [4,4]

        # Make them trainable
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        # We'll store rasterization metadata here
        self.meta = None

        # Optional: visualize initial positions
        self.plot_initial_means()

    def plot_initial_means(self, save_path='initial_means.png'):
        """Plot the initial 2D means in image space."""
        means_np = self.means.detach().cpu().numpy()
        mx = (means_np[:, 0] + 1.0) / 2.0 * self.W
        my = (means_np[:, 1] + 1.0) / 2.0 * self.H

        plt.figure(figsize=(5, 5))
        plt.scatter(mx, my, s=3, c='blue')
        plt.xlim(0, self.W)
        plt.ylim(self.H, 0)
        plt.title("Initial 2D Gaussian Centers")
        plt.savefig(save_path)
        plt.close()
        print(f"Initial means plotted: {save_path}")

    def _get_covs(self):
        """Compute 2x2 covariance from (scales, rotations)."""
        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r,  cos_r], dim=1)
        ], dim=1)  # [N, 2, 2]
        S = torch.diag_embed(self.scales)  # [N, 2, 2]
        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)  # [N,2,2]

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

        if self.meta is not None:
            # Subset attributes for the projected Gaussians
            means2d = self.meta['means2d'].detach().cpu().numpy()   # [M,2]
            conics = self.meta['conics'].detach().cpu().numpy()     # [M,3]
            M = means2d.shape[0]

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

            # Subset rgb, alpha, rotations, scales to match projected Gaussians
            projected_gs = TwoDGaussians(
                means=means2d,                 # [M,2]
                covs=covs2d,                   # [M,2,2]
                rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy()[:M],  # Subset [M,3]
                alpha=torch.sigmoid(self.opacities).detach().cpu().numpy()[:M],  # Subset [M]
                rotations=self.rotations.detach().cpu().numpy()[:M],  # Subset [M]
                scales=self.scales.detach().cpu().numpy()[:M],        # Subset [M,2]
            )
        else:
            projected_gs = None

        return original_gs, projected_gs


    # def train(
    #     self,
    #     iterations: int=2000,
    #     lr: float=0.01,
    #     save_imgs: bool=False,
    #     output_pkl: Optional[str]=None,
    #     scale_reg_weight: float=0.01,  # <--- 追加：スケール正則化の重み
    # ) -> Tuple[TwoDGaussians, Optional[TwoDGaussians], np.ndarray]:
    #     """
    #     Train loop using an orthographic camera, eps2d=0.0, no culling, etc.
    #     """
    #     optimizer = optim.Adam([
    #         self.means, self.scales, self.rotations, self.rgbs, self.opacities
    #     ], lr=lr)
    #     mse_loss_fn = torch.nn.MSELoss()
    #     frames = []
    #     times = [0.0, 0.0]  # [rasterize_time, backward_time]
    #     K = torch.tensor([
    #         [self.focal, 0, self.W/2],
    #         [0, self.focal, self.H/2],
    #         [0, 0, 1],
    #     ], device=self.device)

    #     final_out_img = None

    #     # ここで "1個あたりの目標面積" を計算
    #     # W,H は self.W, self.H
    #     # N = self.num_points
    #     target_area_per_gauss = (self.W * self.H) / float(self.num_points)

    #     for i in range(iterations):
    #         start = time.time()
    #         optimizer.zero_grad()

    #         means_3d = torch.cat([self.means, self.z_means], dim=1)       # [N,3]
    #         scales_3d = torch.cat([self.scales, self.scales_z], dim=1)   # [N,3]

    #         quats = torch.stack([
    #             torch.cos(self.rotations/2),
    #             torch.zeros_like(self.rotations),
    #             torch.zeros_like(self.rotations),
    #             torch.sin(self.rotations/2)
    #         ], dim=1)
    #         quats_norm = quats / quats.norm(dim=-1, keepdim=True)

    #         # Orthographic, no culling:
    #         renders, _, meta = rasterization(
    #             means_3d,
    #             quats_norm,
    #             scales_3d,
    #             torch.sigmoid(self.opacities),
    #             torch.sigmoid(self.rgbs),
    #             self.viewmat[None],   # shape [1,4,4]
    #             K[None],              # shape [1,3,3]
    #             self.W,
    #             self.H,
    #             camera_model="ortho",
    #             near_plane=-1e10,
    #             far_plane=1e10,
    #             radius_clip=-1.0,
    #             rasterize_mode="classic",
    #         )

    #         out_img = renders[0]  # [H,W,3]
    #         if self.device.type == 'cuda':
    #             torch.cuda.synchronize()
    #         times[0] += time.time() - start

    #         # メインのMSEロス
    #         mse_loss = mse_loss_fn(out_img, self.gt_image)

    #         # === 追加: スケール正則化 ===
    #         #   scales[n,0], scales[n,1] -> area_n = scales[n,0] * scales[n,1]
    #         #   これが target_area_per_gauss に近いほどロスが小さい。
    #         area_n = self.scales[:,0] * self.scales[:,1]  # [N,]
    #         scale_reg_loss = torch.mean((area_n - target_area_per_gauss)**2)  # 1個あたりの平均

    #         # 最終ロス
    #         loss = mse_loss + scale_reg_weight * 1e-4 * scale_reg_loss

    #         start = time.time()
    #         loss.backward()
    #         if self.device.type == 'cuda':
    #             torch.cuda.synchronize()
    #         times[1] += time.time() - start

    #         self.losses.append(loss.item())
    #         optimizer.step()

    #         if (i+1) % 50 == 0:
    #             print(f"Iteration {i+1}/{iterations}, MSE={mse_loss.item():.6f}, ScaleReg={scale_reg_loss.item():.6f}, Loss={loss.item():.6f}")

    #         if save_imgs and i % 5 == 0:
    #             frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

    #         final_out_img = out_img.detach().cpu().numpy()
    #         self.meta = meta

    #     if save_imgs and frames:
    #         out_dir = os.path.join(os.getcwd(), "renders")
    #         os.makedirs(out_dir, exist_ok=True)
    #         frames_pil = [Image.fromarray(frame) for frame in frames]
    #         frames_pil[0].save(
    #             f"{out_dir}/training.gif",
    #             save_all=True,
    #             append_images=frames_pil[1:],
    #             optimize=False,
    #             duration=5,
    #             loop=0,
    #         )

    #     print(f"[Timing] Total: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
    #     print(f"[Timing] Per step: R={times[0]/iterations:.6f}, B={times[1]/iterations:.6f}")

    #     original_gs, projected_gs = self.get_gaussians()

    #     if output_pkl is not None:
    #         self.save_gaussians(original_gs, projected_gs, output_pkl)

    #     return original_gs, projected_gs, final_out_img
    
    def train(
        self,
        max_iterations: int = 40000,  # Train to max iterations
        checkpoint_iterations: List[int] = [5000, 10000, 20000, 30000],  # Log metrics at these steps
        lr: float = 0.01,
        save_imgs: bool = False,
        output_pkl: Optional[str] = None,
        gt_np: Optional[np.ndarray] = None,  # Ground truth in NumPy for metrics
        device: torch.device = torch.device('cpu'),  # Device for metrics calculation
    ) -> List[dict]:
        """
        Train loop with intermediate metrics logging.
        Logs PSNR, LPIPS, and SSIM at checkpoint iterations.
        """
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()
        frames = []
        times = [0.0, 0.0]  # [rasterize_time, backward_time]
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)

        final_out_img = None
        results = []  # List of metrics for each checkpoint

        for i in range(max_iterations):
            start = time.time()
            optimizer.zero_grad()

            means_3d = torch.cat([self.means, self.z_means], dim=1)
            scales_3d = torch.cat([self.scales, self.scales_z], dim=1)

            quats = torch.stack([
                torch.cos(self.rotations / 2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations / 2)
            ], dim=1)
            quats_norm = quats / quats.norm(dim=-1, keepdim=True)

            renders, _, meta = rasterization(
                means_3d,
                quats_norm,
                scales_3d,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
            )

            out_img = renders[0]
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[0] += time.time() - start

            loss = mse_loss_fn(out_img, self.gt_image)

            start = time.time()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[1] += time.time() - start

            self.losses.append(loss.item())
            optimizer.step()

            if (i + 1) in checkpoint_iterations:
                print(f"Checkpoint at Iteration {i + 1}/{max_iterations}")
                final_out_img = out_img.detach().cpu().numpy()
                psnr_val = calculate_psnr(final_out_img, gt_np, max_val=1.0)
                lpips_val = calculate_lpips(final_out_img, gt_np, device=device)
                ssim_val = calculate_ssim(final_out_img, gt_np, device=device)
                results.append({
                    "iteration": i + 1,
                    "psnr": psnr_val,
                    "lpips": lpips_val,
                    "ssim": ssim_val,
                })
                print(f"PSNR: {psnr_val:.4f}, LPIPS: {lpips_val:.4f}, SSIM: {ssim_val:.4f}")

            if save_imgs and (i + 1) % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

            self.meta = meta

        # Optionally save final GIF
        if save_imgs and frames:
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames_pil = [Image.fromarray(frame) for frame in frames]
            frames_pil[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames_pil[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        print(f"[Timing] Total: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
        print(f"[Timing] Per step: R={times[0]/max_iterations:.6f}, B={times[1]/max_iterations:.6f}")

        # Save final Gaussians
        original_gs, projected_gs = self.get_gaussians()

        if output_pkl is not None:
            self.save_gaussians(original_gs, projected_gs, output_pkl)

        return results  # List of metrics for all checkpoints


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
        # Modified to use the expected keys
        data = {
            'original_gaussians': original_gs,
            'projected_gaussians': projected_gs,
            'viewmat': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),  # Adding the viewmat used in training
            'K': torch.tensor([
                [self.focal, 0, self.W/2],
                [0, self.focal, self.H/2],
                [0, 0, 1],
            ])  # Adding the camera intrinsics matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Gaussians to: {path}")


def image_path_to_tensor(image_path: Path, device: torch.device = None) -> Tensor:
    """Loads an image and returns shape [H,W,3] in [0..1]."""
    import torchvision.transforms as transforms
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(img).permute(1,2,0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

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
    gt_image = image_path_to_tensor(img_path, device=device)
    gt_np = gt_image.cpu().numpy()

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
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

    gt_image = image_path_to_tensor(img_path, device=device)
    gt_np = gt_image.cpu().numpy()

    results_for_image = []

    # For each npoints, run a single training up to max(iterations_list).
    # The train function uses "checkpoint_iterations=iterations_list" to log at each checkpoint
    for npoints in num_points_list:
        print(f"[GPU {gpu_id}] -> {img_path.name}, npoints={npoints}")
        trainer = SimpleTrainer(gt_image=gt_image, num_points=npoints)

        # Train until the maximum iteration
        all_metrics = trainer.train(
            max_iterations=max(iterations_list),
            checkpoint_iterations=iterations_list,
            lr=lr,
            save_imgs=save_imgs,
            gt_np=gt_np,
            device=device,
        )

        # all_metrics is a list of dictionaries: 
        # [{"iteration": i, "psnr": p, "lpips": l, "ssim": s}, ...]
        for metrics in all_metrics:
            # Gather them into the results_for_image structure
            results_for_image.append({
                "image_name": img_path.name,
                "num_points": npoints,
                "iterations": metrics["iteration"],  # iteration from the checkpoint
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
            # Single image or synthetic
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.img_path:
                gt_image = image_path_to_tensor(Path(args.img_path), device=device)
            else:
                gt_image = generate_synthetic_image(device, height=args.height, width=args.width)

            trainer = SimpleTrainer(gt_image=gt_image, num_points=args.num_points)
            trainer.train(
                iterations=args.iterations,
                lr=args.lr,
                save_imgs=args.save_imgs,
                output_pkl=args.output_path
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
            # Single image or synthetic
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.img_path:
                img_path = Path(args.img_path)
                gt_image = image_path_to_tensor(img_path, device=device)
                image_name = img_path.name
            else:
                gt_image = generate_synthetic_image(device)
                image_name = "synthetic"

            with open(args.output_log, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            for np_ in args.num_points_list:
                for it_ in args.iterations_list:
                    print(f"[EXPERIMENT single] npoints={np_}, iters={it_}")
                    trainer = SimpleTrainer(gt_image=gt_image, num_points=np_)
                    _, _, final_img = trainer.train(iterations=it_, lr=args.lr, save_imgs=args.save_imgs)
                    gt_np = gt_image.cpu().numpy()
                    psnr_val = calculate_psnr(final_img, gt_np, max_val=1.0)
                    lpips_val = calculate_lpips(final_img, gt_np, device=device)
                    ssim_val = calculate_ssim(final_img, gt_np, device=device)

                    with open(args.output_log, "a", newline='') as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writerow({
                            "image_name": image_name,
                            "num_points": np_,
                            "iterations": it_,
                            "psnr": psnr_val,
                            "lpips": lpips_val,
                            "ssim": ssim_val
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
