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

import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor, optim

# These libraries must be installed:
#   gsplat (for rasterization)
#   twodgs (for TwoDGaussians class)
#   lpips (for LPIPS metric)
#   pytorch_msssim (for SSIM metric)
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
    """
    Calculate LPIPS between img and gt.
    img, gt: numpy arrays in [0,1], shape [H,W,3]
    """
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # LPIPS expects inputs in [-1, +1]
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2.0 - 1.0
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = loss_fn_vgg(img_t, gt_t).item()
    return lpips_val


def calculate_ssim(img: np.ndarray, gt: np.ndarray, device: torch.device = torch.device('cpu')) -> float:
    """
    Calculate SSIM between img and gt.
    img, gt: [H,W,3] in [0,1]
    """
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


@dataclass
class ExperimentArgs:
    img_path: Optional[str] = None
    image_folder: Optional[str] = None
    num_points_list: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])
    iterations_list: List[int] = field(default_factory=lambda: [2000, 5000, 10000])
    lr: float = 0.01
    save_imgs: bool = False
    output_log: str = "experiment_results.csv"
    auto_plot: bool = True  # run plot after experiment is done


@dataclass
class PlotArgs:
    csv_path: str = "experiment_results.csv"


class SimpleTrainer:
    """Trains 2D gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = gt_image.device
        print(f"Using device: {self.device}")
        self.gt_image = gt_image
        self.num_points = num_points
        self.losses = []

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize random 2D gaussians"""
        bd = 2
        d = 3

        # X,Y in [-1, +1], random
        self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)  # [N, 2]
        self.scales = torch.rand(self.num_points, 2, device=self.device)  # [N, 2]
        self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi  # [N]
        self.rgbs = torch.rand(self.num_points, d, device=self.device)  # [N, 3]
        self.opacities = torch.ones(self.num_points, device=self.device)  # [N]

        # Z-axis parameters (grad off for scale_z, but let's keep it for demonstration)
        self.z_means = torch.zeros(self.num_points, 1, device=self.device)  # [N,1]
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)  # [N,1]

        # 4x4 extrinsic view matrix
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )  # [4,4]

        # Mark parameters as requiring grad
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        # Optionally, you can visualize the initial 2D positions:
        self.plot_initial_means(save_path='initial_means.png')

        # Will hold the meta data from rasterization
        self.meta = None

    def plot_initial_means(self, save_path: str = 'initial_means.png'):
        """Plot the initial Gaussian centers in the image coordinate space and save."""
        means = self.means.detach().cpu().numpy()
        # transform from [-1,+1] range to pixel coordinates
        means_x = (means[:, 0] + 1.0) / 2.0 * self.W
        means_y = (means[:, 1] + 1.0) / 2.0 * self.H

        plt.figure(figsize=(6, 6))
        plt.scatter(means_x, means_y, s=2, color='blue')
        plt.xlim(0, self.W)
        plt.ylim(self.H, 0)
        plt.title('Initial Gaussian Centers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Initial Gaussian centers plotted: {save_path}")

    def _get_covs(self):
        """Convert scales and rotations to covariance matrices in 2D."""
        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        # R is the rotation matrix
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r,  cos_r], dim=1)
        ], dim=1)  # [N, 2, 2]
        # S is diag(scales)
        S = torch.diag_embed(self.scales)  # [N, 2, 2]
        # covariance = R @ S^2 @ R^T
        # S^2 => S @ S^T but here it's diagonal, so S^2 is straightforward
        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)  # [N, 2, 2]

    def get_gaussians(self) -> Tuple[TwoDGaussians, Optional[TwoDGaussians]]:
        """Return TwoDGaussians object for the original and projected gaussians."""
        with torch.no_grad():
            covs = self._get_covs()

        original_gaussians = TwoDGaussians(
            means=self.means.detach().cpu().numpy(),
            covs=covs.detach().cpu().numpy(),
            rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy(),
            alpha=torch.sigmoid(self.opacities).detach().cpu().numpy(),
            rotations=self.rotations.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy(),
        )

        if self.meta is not None:
            means2d = self.meta['means2d'].detach().cpu().numpy()  # [N,2]
            conics = self.meta['conics'].detach().cpu().numpy()    # [N,3]

            # conics = [A, B, C], which is the *inverse* covariance in 2D
            A = conics[:, 0]
            B = conics[:, 1]
            C = conics[:, 2]

            N = conics.shape[0]
            inv_covs = np.zeros((N, 2, 2))
            inv_covs[:, 0, 0] = A
            inv_covs[:, 0, 1] = B / 2
            inv_covs[:, 1, 0] = B / 2
            inv_covs[:, 1, 1] = C

            covs2d = np.linalg.inv(inv_covs)  # [N,2,2]

            rgbs = torch.sigmoid(self.rgbs).detach().cpu().numpy()  # [N,3]
            alphas = torch.sigmoid(self.opacities).detach().cpu().numpy()  # [N]

            projected_gaussians = TwoDGaussians(
                means=means2d,
                covs=covs2d,
                rgb=rgbs,
                alpha=alphas,
                rotations=self.rotations.detach().cpu().numpy(),
                scales=self.scales.detach().cpu().numpy()
            )
        else:
            projected_gaussians = None

        return original_gaussians, projected_gaussians

    def train(
        self,
        iterations: int = 2000,
        lr: float = 0.01,
        save_imgs: bool = False,
        output_pkl: Optional[str] = None,
    ) -> Tuple[TwoDGaussians, Optional[TwoDGaussians], np.ndarray]:
        """
        Train loop to minimize MSE loss between rasterized Gaussians and gt_image.
        """
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities,
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()
        frames = []  # for saving intermediate frames
        times = [0, 0]  # [rasterization_time, backward_time]
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)

        final_out_img = None

        for iter_idx in range(iterations):
            start = time.time()
            optimizer.zero_grad()

            # Construct 3D means and scales
            means_3d = torch.cat([self.means, self.z_means], dim=1)      # [N,3]
            scales_3d = torch.cat([self.scales, self.scales_z], dim=1)  # [N,3]

            # Convert rotation angle -> quaternion for each gaussian
            quats = torch.stack([
                torch.cos(self.rotations / 2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations / 2)
            ], dim=1)
            quats_normalized = quats / quats.norm(dim=-1, keepdim=True)

            # Rasterize
            renders, _, meta = rasterization(
                means_3d,
                quats_normalized,
                scales_3d,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],  # [1,4,4]
                K[None],             # [1,3,3]
                self.W,
                self.H,
            )

            out_img = renders[0]  # shape [H,W,3]
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[0] += time.time() - start

            # compute MSE loss
            loss = mse_loss_fn(out_img, self.gt_image)

            # Backprop
            start = time.time()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[1] += time.time() - start

            self.losses.append(loss.item())
            optimizer.step()

            if (iter_idx + 1) % 50 == 0:
                print(f"Iteration {iter_idx + 1}/{iterations}, Loss: {loss.item():.6f}")

            if save_imgs and (iter_idx % 5 == 0):
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

            final_out_img = out_img.detach().cpu().numpy()

        # optionally save a gif
        if save_imgs and len(frames) > 0:
            frames_pil = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames_pil[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames_pil[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        print(f"[Timing] Total: Rasterization: {times[0]:.3f}s, Backward: {times[1]:.3f}s")
        print(f"[Timing] Per step: R {times[0]/iterations:.5f}s, B {times[1]/iterations:.5f}s")

        self.meta = meta
        original_gaussians, projected_gaussians = self.get_gaussians()

        # If user wants to save PKL with fitted gaussians
        if output_pkl is not None:
            self.save_gaussians(original_gaussians, projected_gaussians, output_pkl)

        return original_gaussians, projected_gaussians, final_out_img

    def plot_loss(self, save_path: str = 'loss_curve.png'):
        """Save a simple plot of the MSE loss over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved: {save_path}")

    @staticmethod
    def save_gaussians(original_gs: TwoDGaussians, projected_gs: Optional[TwoDGaussians], path: str):
        """Save the fitted gaussians as a pickle file."""
        import pickle
        data = {
            'original': original_gs,
            'projected': projected_gs,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Gaussians to: {path}")


def image_path_to_tensor(image_path: Path, device: torch.device = None) -> Tensor:
    """Load an image and convert to a torch Tensor [H,W,3] in [0,1]."""
    import torchvision.transforms as transforms

    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]  # shape [H,W,3]
    if device is not None:
        img_tensor = img_tensor.to(device)
    return img_tensor


def plot_results_from_csv(csv_path: str) -> None:
    """
    Plot the results from a CSV file that contains columns:
      image_name, num_points, iterations, psnr, lpips, ssim
    Creates separate line plots for PSNR, LPIPS, and SSIM vs. num_points for each iteration count.
    """
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "image_name": row.get("image_name", "unknown"),
                "num_points": int(row["num_points"]),
                "iterations": int(row["iterations"]),
                "psnr": float(row["psnr"]),
                "lpips": float(row["lpips"]),
                "ssim": float(row["ssim"]),
            })

    # Group by iteration -> list of {image_name, num_points, psnr, lpips, ssim...}
    results_by_iterations = defaultdict(list)
    for entry in data:
        results_by_iterations[entry["iterations"]].append(entry)

    # Sort each list by num_points
    for iters, vals in results_by_iterations.items():
        vals.sort(key=lambda x: x["num_points"])

    # Plot PSNR vs num_points
    plt.figure(figsize=(10, 6))
    for iters, vals in results_by_iterations.items():
        x = [v["num_points"] for v in vals]
        y = [v["psnr"] for v in vals]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.title("PSNR vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("PSNR (Higher is better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_num_points.png")
    plt.close()
    print("PSNR plot saved: psnr_vs_num_points.png")

    # Plot LPIPS vs num_points
    plt.figure(figsize=(10, 6))
    for iters, vals in results_by_iterations.items():
        x = [v["num_points"] for v in vals]
        y = [v["lpips"] for v in vals]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.title("LPIPS vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("LPIPS (Lower is better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lpips_vs_num_points.png")
    plt.close()
    print("LPIPS plot saved: lpips_vs_num_points.png")

    # Plot SSIM vs num_points
    plt.figure(figsize=(10, 6))
    for iters, vals in results_by_iterations.items():
        x = [v["num_points"] for v in vals]
        y = [v["ssim"] for v in vals]
        plt.plot(x, y, marker='o', label=f"iters={iters}")
    plt.title("SSIM vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("SSIM (Higher is better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ssim_vs_num_points.png")
    plt.close()
    print("SSIM plot saved: ssim_vs_num_points.png")


def load_images_from_folder(folder: Path, device: torch.device) -> List[Path]:
    """Return a list of all image paths (jpg, png) in the folder."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts:
            image_paths.append(f)
    image_paths.sort()
    return image_paths


def generate_synthetic_image(device: torch.device, height: int = 256, width: int = 256) -> Tensor:
    """
    Generate a 256x256 image where top-left is red, bottom-right is blue, and the rest is white.
    shape: [H,W,3], range [0,1]
    """
    gt_image = torch.ones((height, width, 3), device=device, dtype=torch.float32)
    gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=device)
    gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0], device=device)
    return gt_image


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
    parser_train.add_argument('--output_path', type=str, default='fitted_gaussians.pkl',
                              help='Default PKL path (if single image). Otherwise automatically named for each image.')

    # --- Experiment subcommand ---
    parser_experiment = subparsers.add_parser('experiment', help='Run multiple training sessions (grid search) with different parameters.')
    parser_experiment.add_argument('--img_path', type=str, default=None, help='Path to a single image')
    parser_experiment.add_argument('--image_folder', type=str, default=None, help='Folder of images to run experiment on')
    parser_experiment.add_argument('--num_points_list', type=int, nargs='+', default=[100, 200, 500, 1000, 2000],
                                   help='List of number of Gaussian points to experiment with')
    parser_experiment.add_argument('--iterations_list', type=int, nargs='+', default=[2000, 5000, 10000],
                                   help='List of iteration counts to experiment with')
    parser_experiment.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser_experiment.add_argument('--save_imgs', action='store_true', help='Save rendered images as GIF')
    parser_experiment.add_argument('--output_log', type=str, default='experiment_results.csv', help='Path to save experiment results CSV')
    parser_experiment.add_argument('--auto_plot', action='store_true', help='Automatically generate plots after experiments')

    # --- Plot subcommand ---
    parser_plot = subparsers.add_parser('plot', help='Plot results from a CSV file')
    parser_plot.add_argument('--csv_path', type=str, default='experiment_results.csv', help='Path to the CSV file containing experiment results')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == 'train':
        # Handle single or multiple training sessions
        if args.image_folder is not None:
            # Train on every image in the folder
            image_paths = load_images_from_folder(args.image_folder, device=device)
            if len(image_paths) == 0:
                print(f"No valid images found in {args.image_folder}")
                sys.exit(1)
            
            # workplace/dataに出力フォルダを作成
            current_dir = Path(__file__).parent  # examples directory
            workspace_dir = current_dir.parent.parent  # workplace directory
            data_dir = workspace_dir / "data"
            
            # 入力フスからscanXXの部分を抽出
            input_folder = Path(args.image_folder)
            scan_name = input_folder.parent.name  # 'scan63'のような部分を取得
            
            # 出力フォルダを作成 (例: data/scan63_results/)
            output_folder = data_dir / f"{scan_name}_results"
            output_folder.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_folder}")

            for img_path in image_paths:
                print(f"\n[TRAIN] Processing {img_path.name}")
                gt_image = image_path_to_tensor(img_path, device=device)
                trainer = SimpleTrainer(gt_image=gt_image, num_points=args.num_points)
                
                # PKLファイルを出力フォルダ内に保存
                out_name = output_folder / f"{img_path.stem}_fitted_gaussians.pkl"
                trainer.train(
                    iterations=args.iterations,
                    lr=args.lr,
                    save_imgs=args.save_imgs,
                    output_pkl=str(out_name)
                )
                
                # ロス曲線も同じフォルダに保存
                loss_path = output_folder / f"{img_path.stem}_loss_curve.png"
                trainer.plot_loss(save_path=str(loss_path))

        else:
            # Single image or synthetic
            if args.img_path:
                img_path = args.img_path
                gt_image = image_path_to_tensor(img_path, device=device)
            else:
                # synthetic
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
        # Handle experiments with multiple images or a single image
        if args.image_folder is not None:
            # Loop over all images in folder
            folder = Path(args.image_folder)
            image_paths = load_images_from_folder(folder, device=device)
            if len(image_paths) == 0:
                print(f"No valid images found in {folder}")
                sys.exit(1)

            # Prepare CSV
            fieldnames = ["image_name", "num_points", "iterations", "psnr", "lpips", "ssim"]
            with open(args.output_log, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            # We'll store the results in memory to compute averages
            aggregated_results = defaultdict(list)  # key=(num_points, iters), val=list of (psnr, lpips, ssim)

            for img_path in image_paths:
                print(f"\n[EXPERIMENT] {img_path.name}")
                gt_image = image_path_to_tensor(img_path, device=device)

                for npoints in args.num_points_list:
                    for iters in args.iterations_list:
                        print(f"  -> num_points={npoints}, iterations={iters}")
                        trainer = SimpleTrainer(gt_image=gt_image, num_points=npoints)
                        _, _, final_out_img = trainer.train(
                            iterations=iters,
                            lr=args.lr,
                            save_imgs=args.save_imgs,
                            output_pkl=None  # not saving PKL in experiment by default
                        )
                        gt_np = gt_image.detach().cpu().numpy()
                        psnr_val = calculate_psnr(final_out_img, gt_np, max_val=1.0)
                        lpips_val = calculate_lpips(final_out_img, gt_np, device=trainer.device)
                        ssim_val = calculate_ssim(final_out_img, gt_np, device=trainer.device)

                        # Write to CSV
                        with open(args.output_log, "a", newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow({
                                "image_name": img_path.name,
                                "num_points": npoints,
                                "iterations": iters,
                                "psnr": psnr_val,
                                "lpips": lpips_val,
                                "ssim": ssim_val,
                            })

                        # For averaging
                        aggregated_results[(npoints, iters)].append((psnr_val, lpips_val, ssim_val))

            # After all images, compute average for each (num_points, iters)
            avg_results_path = Path(args.output_log).parent / ("experiment_results_averaged.csv")
            with open(avg_results_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["num_points", "iterations", "psnr_mean", "lpips_mean", "ssim_mean", "count"])
                for (npoints, iters), vals in aggregated_results.items():
                    vals = np.array(vals)  # shape [count, 3]
                    psnr_mean, lpips_mean, ssim_mean = vals.mean(axis=0)
                    writer.writerow([npoints, iters, psnr_mean, lpips_mean, ssim_mean, len(vals)])
            print(f"Average results saved to: {avg_results_path}")

            # auto plot if requested
            if args.auto_plot:
                plot_results_from_csv(args.output_log)

        else:
            # Single image or synthetic
            if args.img_path:
                img_path = Path(args.img_path)
                gt_image = image_path_to_tensor(img_path, device=device)
            else:
                # synthetic
                gt_image = generate_synthetic_image(device)

            results = []
            fieldnames = ["image_name","num_points","iterations","psnr","lpips","ssim"]
            with open(args.output_log, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            image_name = args.img_path if args.img_path else "synthetic"
            for npoints in args.num_points_list:
                for iters in args.iterations_list:
                    print(f"[EXPERIMENT single] num_points={npoints}, iterations={iters}")
                    trainer = SimpleTrainer(gt_image=gt_image, num_points=npoints)
                    _, _, final_out_img = trainer.train(iterations=iters, lr=args.lr, save_imgs=args.save_imgs)
                    gt_np = gt_image.detach().cpu().numpy()
                    psnr_val = calculate_psnr(final_out_img, gt_np, max_val=1.0)
                    lpips_val = calculate_lpips(final_out_img, gt_np, device=trainer.device)
                    ssim_val = calculate_ssim(final_out_img, gt_np, device=trainer.device)

                    results.append((npoints, iters, psnr_val, lpips_val, ssim_val))

                    with open(args.output_log, "a", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow({
                            "image_name": image_name,
                            "num_points": npoints,
                            "iterations": iters,
                            "psnr": psnr_val,
                            "lpips": lpips_val,
                            "ssim": ssim_val
                        })

            print(f"Results saved to {args.output_log}")
            if args.auto_plot:
                plot_results_from_csv(args.output_log)

    elif args.command == 'plot':
        # Plot from CSV
        plot_results_from_csv(args.csv_path)

    else:
        print("Unknown command. Use 'train', 'experiment', or 'plot'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
