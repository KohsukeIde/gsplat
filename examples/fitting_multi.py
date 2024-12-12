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

from gsplat import rasterization  # gsplatが正しくインストールされていることを確認
from twodgs import TwoDGaussians  # twodgsが正しくインストールされていることを確認

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
    iterations: int = 2000
    lr: float = 0.01
    output_path: str = 'fitted_gaussians.pkl'


@dataclass
class ExperimentArgs:
    img_path: Optional[str] = None
    num_points_list: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])
    iterations_list: List[int] = field(default_factory=lambda: [2000, 5000, 10000])
    lr: float = 0.01
    save_imgs: bool = False
    output_log: str = "experiment_results.csv"
    auto_plot: bool = True  # 実験終了後に自動的にプロット


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
        print(f"{self.device=}")
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

        self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)  # [-1,1)
        self.scales = torch.rand(self.num_points, 2, device=self.device)  # scales for x and y
        self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi  # rotation in radians
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device)

        # Z軸に関するパラメータを追加（grad off）
        self.z_means = torch.zeros(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        self.meta = None

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )  # [4,4]
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        self.plot_initial_means(save_path='initial_means.png')

    def plot_initial_means(self, save_path: str = 'initial_means.png'):
        """Plot the initial Gaussian centers in image coordinates and save."""
        means = self.means.detach().cpu().numpy()
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
        print(f"Initial Gaussian centers plotted and saved to {save_path}")

    def _get_covs(self):
        """Convert scales and rotations to covariance matrices"""
        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r, cos_r], dim=1)
        ], dim=1)  # [num_points, 2, 2]
        S = torch.diag_embed(self.scales)  # [num_points, 2, 2]
        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)  # [num_points, 2, 2]

    def get_gaussians(self) -> Tuple[TwoDGaussians, Optional[TwoDGaussians]]:
        # 元のガウス分布を取得
        with torch.no_grad():
            covs = self._get_covs()
        original_gaussians = TwoDGaussians(
            means=self.means.detach().cpu().numpy(),
            covs=covs.detach().cpu().numpy(),
            rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy(),
            alpha=torch.sigmoid(self.opacities).detach().cpu().numpy(),
            rotations=self.rotations.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy()
        )

        # 投影後のガウス分布を取得
        if self.meta is not None:
            means2d = self.meta['means2d'].detach().cpu().numpy()  # [N, 2]
            conics = self.meta['conics'].detach().cpu().numpy()    # [N, 3]

            # conics から逆共分散行列を構築
            A = conics[:, 0]
            B = conics[:, 1]
            C = conics[:, 2]

            N = conics.shape[0]
            inv_covs = np.zeros((N, 2, 2))
            inv_covs[:, 0, 0] = A
            inv_covs[:, 0, 1] = B / 2
            inv_covs[:, 1, 0] = B / 2
            inv_covs[:, 1, 1] = C

            # 逆共分散行列を反転して共分散行列を取得
            covs2d = np.linalg.inv(inv_covs)  # 形状: [N, 2, 2]

            # rgbs と opacities を取得
            rgbs = torch.sigmoid(self.rgbs).detach().cpu().numpy()    # [N, 3]
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
    ) -> Tuple[TwoDGaussians, Optional[TwoDGaussians], np.ndarray]:
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities,
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()
        frames = []
        times = [0, 0]  # [rasterization, backward]
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)

        final_out_img = None

        for iter in range(iterations):
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
            quats_normalized = quats / quats.norm(dim=-1, keepdim=True)

            renders, _, meta = rasterization(
                means_3d,
                quats_normalized,
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
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times[1] += time.time() - start

            self.losses.append(loss.item())
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

            final_out_img = out_img.detach().cpu().numpy()

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

        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )

        self.meta = meta
        original_gaussians, projected_gaussians = self.get_gaussians()

        # save_gaussians の呼び出しをコメントアウト
        # self.save_gaussians(original_gaussians, projected_gaussians, 'fitted_gaussians.pkl')

        return original_gaussians, projected_gaussians, final_out_img

    def plot_loss(self, save_path: str = 'loss_curve.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")


def image_path_to_tensor(image_path: Path) -> Tensor:
    import torchvision.transforms as transforms

    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def plot_results_from_csv(csv_path: str) -> None:
    """
    Plot the results from a CSV file that contains columns:
    num_points,iterations,psnr,lpips,ssim
    We'll create a line plot of PSNR, LPIPS, and SSIM for various num_points at fixed iterations.
    """
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "num_points": int(row["num_points"]),
                "iterations": int(row["iterations"]),
                "psnr": float(row["psnr"]),
                "lpips": float(row["lpips"]),
                "ssim": float(row["ssim"])
            })

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
        plt.plot(x, y, marker='o', label=f"iterations={iters}")
    plt.title("PSNR vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_num_points.png")
    plt.close()
    print("PSNR plot saved to psnr_vs_num_points.png")

    # Plot LPIPS vs num_points
    plt.figure(figsize=(10, 6))
    for iters, vals in results_by_iterations.items():
        x = [v["num_points"] for v in vals]
        y = [v["lpips"] for v in vals]
        plt.plot(x, y, marker='o', label=f"iterations={iters}")
    plt.title("LPIPS vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("LPIPS (lower is better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("lpips_vs_num_points.png")
    plt.close()
    print("LPIPS plot saved to lpips_vs_num_points.png")

    # Plot SSIM vs num_points
    plt.figure(figsize=(10, 6))
    for iters, vals in results_by_iterations.items():
        x = [v["num_points"] for v in vals]
        y = [v["ssim"] for v in vals]
        plt.plot(x, y, marker='o', label=f"iterations={iters}")
    plt.title("SSIM vs num_points")
    plt.xlabel("num_points")
    plt.ylabel("SSIM (higher is better)")
    plt.grid(True)
    plt.legend()
    plt.savefig("ssim_vs_num_points.png")
    plt.close()
    print("SSIM plot saved to ssim_vs_num_points.png")


def main():
    parser = argparse.ArgumentParser(description="Fitting 2D Gaussians to an Image")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # Train subcommand
    parser_train = subparsers.add_parser('train', help='Run a single training session')
    parser_train.add_argument('--height', type=int, default=256, help='Height of the ground truth image')
    parser_train.add_argument('--width', type=int, default=256, help='Width of the ground truth image')
    parser_train.add_argument('--num_points', type=int, default=200, help='Number of Gaussian points')
    parser_train.add_argument('--save_imgs', action='store_true', help='Save rendered images as GIF')
    parser_train.add_argument('--img_path', type=Path, default=None, help='Path to the ground truth image')
    parser_train.add_argument('--iterations', type=int, default=2000, help='Number of training iterations')
    parser_train.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser_train.add_argument('--output_path', type=str, default='fitted_gaussians.pkl', help='Path to save fitted Gaussians')

    # Experiment subcommand
    parser_experiment = subparsers.add_parser('experiment', help='Run multiple training sessions with different parameters')
    parser_experiment.add_argument('--img_path', type=str, default=None, help='Path to the ground truth image')
    parser_experiment.add_argument('--num_points_list', type=int, nargs='+', default=[100, 200, 500, 1000, 2000], help='List of number of Gaussian points to experiment with')
    parser_experiment.add_argument('--iterations_list', type=int, nargs='+', default=[2000, 5000, 10000], help='List of iteration counts to experiment with')
    parser_experiment.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser_experiment.add_argument('--save_imgs', action='store_true', help='Save rendered images as GIF')
    parser_experiment.add_argument('--output_log', type=str, default='experiment_results.csv', help='Path to save experiment results CSV')
    parser_experiment.add_argument('--auto_plot', action='store_true', help='Automatically generate plots after experiments')

    # Plot subcommand
    parser_plot = subparsers.add_parser('plot', help='Plot results from a CSV file')
    parser_plot.add_argument('--csv_path', type=str, default='experiment_results.csv', help='Path to the CSV file containing experiment results')

    args = parser.parse_args()

    if args.command == 'train':
        # Handle single training session
        if args.img_path:
            gt_image = image_path_to_tensor(args.img_path)
        else:
            gt_image = torch.ones((args.height, args.width, 3), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 1.0
            gt_image[: args.height // 2, : args.width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=gt_image.device)
            gt_image[args.height // 2 :, args.width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0], device=gt_image.device)

        trainer = SimpleTrainer(gt_image=gt_image, num_points=args.num_points)
        original_gaussians, projected_gaussians, final_out_img = trainer.train(
            iterations=args.iterations,
            lr=args.lr,
            save_imgs=args.save_imgs,
        )

        print(f"Fitted {original_gaussians.means.shape[0]} Gaussians")

        trainer.plot_loss(save_path="loss_curve.png")

    elif args.command == 'experiment':
        # Handle multiple training sessions
        if args.img_path:
            gt_image = image_path_to_tensor(Path(args.img_path)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            height, width = 256, 256
            gt_image = torch.ones((height, width, 3), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 1.0
            gt_image[:height//2, :width//2, :] = torch.tensor([1.0,0.0,0.0], device=gt_image.device)
            gt_image[height//2:, width//2:, :] = torch.tensor([0.0,0.0,1.0], device=gt_image.device)

        results = []
        for npoints in args.num_points_list:
            for iters in args.iterations_list:
                print(f"Starting experiment with num_points={npoints}, iterations={iters}")
                trainer = SimpleTrainer(gt_image=gt_image, num_points=npoints)
                original_gaussians, projected_gaussians, final_out_img = trainer.train(
                    iterations=iters, lr=args.lr, save_imgs=args.save_imgs,
                )
                gt_np = gt_image.detach().cpu().numpy()
                psnr_val = calculate_psnr(final_out_img, gt_np, max_val=1.0)
                lpips_val = calculate_lpips(final_out_img, gt_np, device=trainer.device)
                ssim_val = calculate_ssim(final_out_img, gt_np, device=trainer.device)
                results.append((npoints, iters, psnr_val, lpips_val, ssim_val))
                print(f"num_points={npoints}, iterations={iters}, PSNR={psnr_val}, LPIPS={lpips_val}, SSIM={ssim_val}")

        # Save results to CSV
        with open(args.output_log, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["num_points","iterations","psnr","lpips","ssim"])
            for r in results:
                writer.writerow(r)
        print(f"Results saved to {args.output_log}")

        # Auto plot if enabled
        if args.auto_plot:
            plot_results_from_csv(args.output_log)

    elif args.command == 'plot':
        # Handle plotting
        plot_results_from_csv(args.csv_path)

    else:
        print("Unknown command. Use 'train', 'experiment', or 'plot'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
