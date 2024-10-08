import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization
from twodgs import TwoDGaussians  # 正しいパスに変更してください

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

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        # self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Initialize random 2D gaussians"""
        self.means = torch.rand(self.num_points, 2, device=self.device) * torch.tensor([self.W, self.H], device=self.device)
        self.scales = torch.rand(self.num_points, 2, device=self.device) * 10  # scales for x and y
        self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi  # rotation in radians
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device) * 0.1

        # Z軸に関するパラメータを追加
        self.z_means = torch.zeros(self.num_points, 1, device=self.device, requires_grad=True)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device, requires_grad=True)

  

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        ) # [4, 4]
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

    def get_gaussians(self) -> TwoDGaussians:
        with torch.no_grad():
            covs = self._get_covs()
        return TwoDGaussians(
            means=self.means.detach().cpu().numpy(),
            covs=covs.detach().cpu().numpy(),
            rgb=self.rgbs.detach().cpu().numpy(),
            alpha=torch.sigmoid(self.opacities).detach().cpu().numpy()
        )

    def train(
        self,
        iterations: int = 2000,
        lr: float = 0.01,
        save_imgs: bool = False,
    ):
        # オプティマイザにZ軸のパラメータも含める
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities, self.z_means, self.scales_z
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
                print(f"means_3d shape: {means_3d.shape}")                    # [N, 3]
                print(f"quats_normalized shape: {quats_normalized.shape}")    # [N, 4]
                print(f"scales_3d shape: {scales_3d.shape}")                  # [N, 3]
                print(f"K shape: {K.shape}")                              # [3, 3]
                print(f"viewmat shape: {self.viewmat.shape}")                  # [4, 4]
            renders, _, _ = rasterization(
                means_3d,                # [N, 3]
                quats_normalized,        # [N, 4]
                scales_3d,               # [N, 3]
                torch.sigmoid(self.opacities),  # [N]
                torch.sigmoid(self.rgbs),       # [N, 3]
                self.viewmat[None],         # [1, 4, 4]
                K[None],                  # [1, 3, 3]
                self.W,
                self.H,
            )
            out_img = renders[0]
            torch.cuda.synchronize()
            times[0] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            # 勾配の確認（最初の5イテレーションのみ）
            if iter < 5:
                print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")
                print(f"means grad mean: {self.means.grad.abs().mean().item() if self.means.grad is not None else 'None'}")
                print(f"scales grad mean: {self.scales.grad.abs().mean().item() if self.scales.grad is not None else 'None'}")
                print(f"rotations grad mean: {self.rotations.grad.abs().mean().item() if self.rotations.grad is not None else 'None'}")
                print(f"rgbs grad mean: {self.rgbs.grad.abs().mean().item() if self.rgbs.grad is not None else 'None'}")
                print(f"opacities grad mean: {self.opacities.grad.abs().mean().item() if self.opacities.grad is not None else 'None'}")
                print(f"z_means grad mean: {self.z_means.grad.abs().mean().item() if self.z_means.grad is not None else 'None'}")
                print(f"scales_z grad mean: {self.scales_z.grad.abs().mean().item() if self.scales_z.grad is not None else 'None'}")

                # sys.exit()

            # パラメータ更新
            optimizer.step()

            # 損失出力
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            # 100イテレーションごとにレンダリング画像の統計を出力
            if iter % 100 == 0:
                print(f"Rendered image mean: {out_img.mean().item()}, std: {out_img.std().item()}")

            # レンダリング画像の保存
            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

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

        return self.get_gaussians()

def image_path_to_tensor(image_path: Path) -> Tensor:
    img = Image.open(image_path).convert('RGB')
    return torch.FloatTensor(np.array(img)) / 255.0

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 5000,
    lr: float = 0.01,
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * 1.0
        # 左上と右下を赤と青に設定
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    fitted_gaussians = trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )
    
    print(f"Fitted {fitted_gaussians.k} Gaussians")

if __name__ == "__main__":
    tyro.cli(main)

