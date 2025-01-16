#!/usr/bin/env python

import math
import os
import sys
import time
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import torch
import tyro
from PIL import Image
import matplotlib.pyplot as plt
from torch import Tensor, optim

from gsplat import rasterization
from twodgs import TwoDGaussians


class SimpleTrainer:
    """Trains 2D gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.device=}")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.losses = []

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        # self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize random 2D gaussians"""
        bd = 2
        d = 3

        self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)
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
        )  # [4, 4]
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

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

    # def get_gaussians(self) -> TwoDGaussians:
    #     with torch.no_grad():
    #         covs = self._get_covs()
    #     return TwoDGaussians(
    #         means=self.means.detach().cpu().numpy(),
    #         covs=covs.detach().cpu().numpy(),
    #         rgb=torch.sigmoid(self.rgbs).detach().cpu().numpy(),
    #         alpha=torch.sigmoid(self.opacities).detach().cpu().numpy(),
    #         rotations=self.rotations.detach().cpu().numpy(),
    #         scales=self.scales.detach().cpu().numpy()
    #     )

    def get_gaussians(self) -> tuple:
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

    # def save_gaussians(self, gaussians: TwoDGaussians, file_path: str):
    #     print(f"Saving means: {gaussians.means[:5]}")
    #     print(f"Saving covs: {gaussians.covs[:5]}")
    #     print(f"Saving rgbs: {gaussians.rgb[:5]}")
    #     print(f"Saving alphas: {gaussians.alpha[:5]}")
    #     data = {
    #         "gaussians": gaussians,
    #         "viewmat": self.viewmat.detach().cpu().numpy(),
    #         "K": torch.tensor([
    #             [self.focal, 0, self.W / 2],
    #             [0, self.focal, self.H / 2],
    #             [0, 0, 1],
    #         ], device=self.device).detach().cpu().numpy()
    #     }
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(data, f)
    #     print(f"Saved gaussians and camera parameters to {file_path}")

    def save_gaussians(self, original_gaussians: TwoDGaussians, projected_gaussians: TwoDGaussians, file_path: str):
        print(f"Saving original means: {original_gaussians.means[:5]}")
        print(f"Saving original covs: {original_gaussians.covs[:5]}")
        print(f"Saving projected means: {projected_gaussians.means[:5]}")
        print(f"Saving projected covs: {projected_gaussians.covs[:5]}")
        print(f"Saving rgbs: {original_gaussians.rgb[:5]}")
        print(f"Saving alphas: {original_gaussians.alpha[:5]}")
        data = {
            "original_gaussians": original_gaussians,
            "projected_gaussians": projected_gaussians,
            "viewmat": self.viewmat.detach().cpu().numpy(),
            "K": torch.tensor([
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ], device=self.device).detach().cpu().numpy()
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved gaussians and camera parameters to {file_path}")

    def train(
        self,
        iterations: int = 2000,
        lr: float = 0.01,
        save_imgs: bool = False,
    ):
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities,
        ], lr=lr)
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)  # [3, 3]

        for iter in range(iterations):
            start = time.time()
            optimizer.zero_grad()

            # 2Dパラメータを3Dに拡張
            means_3d = torch.cat([self.means, self.z_means], dim=1)  # [N, 3]
            scales_3d = torch.cat([self.scales, self.scales_z], dim=1)  # [N, 3]

            # 2D回転角度からクォータニオンを計算（Z軸周りの回転）
            quats = torch.stack([
                torch.cos(self.rotations / 2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations / 2)
            ], dim=1)  # [N, 4]
            quats_normalized = quats / quats.norm(dim=-1, keepdim=True)  # [N, 4]

            if iter == 0:
                print(f"means_3d shape: {means_3d.shape}")  # [N, 3]
                print(f"quats_normalized shape: {quats_normalized.shape}")  # [N, 4]
                print(f"scales_3d shape: {scales_3d.shape}")  # [N, 3]
                print(f"K shape: {K.shape}")  # [3, 3]
                print(f"viewmat shape: {self.viewmat.shape}")  # [4, 4]

            renders, _, meta = rasterization(
                means_3d,                # [N, 3]
                quats_normalized,        # [N, 4]
                scales_3d,               # [N, 3]
                torch.sigmoid(self.opacities),  # [N]
                torch.sigmoid(self.rgbs),       # [N, 3]
                self.viewmat[None],      # [1, 4, 4]
                K[None],                 # [1, 3, 3]
                self.W,
                self.H,
            )

            out_img = renders[0]

            # === ここで [0, 1] にクリッピングを追加 ===
            out_img = torch.clamp(out_img, 0.0, 1.0)

            torch.cuda.synchronize()
            times[0] += time.time() - start

            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            self.losses.append(loss.item())

            # 勾配の確認（first 5iter）
            if iter < 5:
                print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
                print(f"means grad mean: {self.means.grad.abs().mean().item() if self.means.grad is not None else 'None'}")
                print(f"scales grad mean: {self.scales.grad.abs().mean().item() if self.scales.grad is not None else 'None'}")
                print(f"rotations grad mean: {self.rotations.grad.abs().mean().item() if self.rotations.grad is not None else 'None'}")
                print(f"rgbs grad mean: {self.rgbs.grad.abs().mean().item() if self.rgbs.grad is not None else 'None'}")
                print(f"opacities grad mean: {self.opacities.grad.abs().mean().item() if self.opacities.grad is not None else 'None'}")
                print(f"z_means grad mean: {self.z_means.grad.abs().mean().item() if self.z_means.grad is not None else 'None'}")
                print(f"scales_z grad mean: {self.scales_z.grad.abs().mean().item() if self.scales_z.grad is not None else 'None'}")

            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if iter % 100 == 0:
                print(f"Rendered image mean: {out_img.mean().item()}, std: {out_img.std().item()}")

            if save_imgs and iter % 5 == 0:
                # クリップ済み out_img でフレームを作成
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

            self.meta = meta

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

        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )
        self.meta = meta
        original_gaussians, projected_gaussians = self.get_gaussians()

        self.save_gaussians(original_gaussians, projected_gaussians, 'fitted_gaussians.pkl')
        return original_gaussians, projected_gaussians

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


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 16,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 5000,
    lr: float = 0.01,
    output_path: str = 'fitted_gaussians.pkl',
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 1.0
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    original_gaussians, projected_gaussians = trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )

    print(f"Fitted {original_gaussians.k} Gaussians")

    trainer.plot_loss(save_path="loss_curve.png")


if __name__ == "__main__":
    tyro.cli(main)
