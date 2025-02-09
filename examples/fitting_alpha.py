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
    rgb = t[..., :3]            # [H,W,3]
    alpha = t[..., 3:]          # [H,W,1]

    if device is not None:
        rgb = rgb.to(device)
        alpha = alpha.to(device)

    return rgb, alpha


class SimpleTrainer:
    """Trains 2D gaussians to fit an RGBA image, ignoring alpha=0 backgrounds."""

    def __init__(
        self,
        gt_rgb: Tensor,       # [H,W,3]
        alpha_mask: Tensor,   # [H,W,1], 0=背景,1=前景
        num_points: int = 2000,
    ):
        self.device = gt_rgb.device
        print(f"{self.device=}")

        # (1) GT画像: 背景黒にするため rgb *= alpha
        #    こうすると背景ピクセルは (0,0,0) になる
        self.gt_image = gt_rgb * alpha_mask.expand(-1, -1, 3)  # shape [H,W,3]

        self.alpha_mask = alpha_mask  # 後でサンプリングや学習時のマスクに使う
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
            # alpha>0.5 の座標を取る (例)
            foreground_indices = torch.nonzero(mask_2d > 0.5, as_tuple=False)  # [K,2], (y,x)

            K = foreground_indices.shape[0]
            if K < self.num_points:
                print("Warning: foreground has fewer pixels than num_points. Some gaussians will overlap.")

            chosen = torch.randint(0, K, size=(self.num_points,), device=self.device)
            chosen_xy = foreground_indices[chosen]  # [N,2], (y_i, x_i)

            # 画像座標 [y,x] → 正規化座標 [-1,+1]
            #  x_norm = (x / (W-1))*2 - 1
            #  y_norm = (y / (H-1))*2 - 1
            y = chosen_xy[:, 0].float()
            x = chosen_xy[:, 1].float()
            x_norm = (x / (self.W - 1))*2.0 - 1.0
            y_norm = (y / (self.H - 1))*2.0 - 1.0
            self.means = torch.stack([x_norm, y_norm], dim=-1)  # [N,2]
        else:
            # 従来通り [-1,+1] にランダム
            bd = 2
            self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)

        # スケール・回転・RGB・opacities は従来通り
        self.scales = torch.rand(self.num_points, 2, device=self.device)
        self.rotations = torch.rand(self.num_points, device=self.device) * 2*math.pi
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.opacities = torch.ones(self.num_points, device=self.device)

        # Zパラメータ (固定or自由)
        self.z_means = torch.zeros(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        # ビュー行列 (固定)
        self.viewmat = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device)

        # 学習パラメータに指定
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True

        self.viewmat.requires_grad = False

        self.meta = None

    def _get_covs(self):
        """Compute 2x2 covariance from scales, rotations."""
        cos_r = torch.cos(self.rotations)
        sin_r = torch.sin(self.rotations)
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r,  cos_r], dim=1)
        ], dim=1)  # [N,2,2]
        S = torch.diag_embed(self.scales)  # [N,2,2]
        return R @ S @ S.transpose(1,2) @ R.transpose(1,2)

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

        if self.meta is not None:
            means2d = self.meta['means2d'].detach().cpu().numpy()   # [M,2]
            conics = self.meta['conics'].detach().cpu().numpy()     # [M,3]
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

            # 同数だけ取り出す(本当はmetaに対応づけが必要だが、単純化)
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
        else:
            projected_gs = None

        return original_gs, projected_gs

    def save_gaussians(self, orig_gs: TwoDGaussians, proj_gs: TwoDGaussians, path: str):
        import pickle
        data = {
            "original_gaussians": orig_gs,
            "projected_gaussians": proj_gs,
            "viewmat": self.viewmat.detach().cpu().numpy(),
            "K": torch.tensor([
                [self.focal, 0, self.W/2],
                [0, self.focal, self.H/2],
                [0, 0, 1],
            ], device=self.device).detach().cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved gaussians to {path}")

    def train(
        self,
        iterations: int = 2000,
        lr: float = 0.01,
        save_imgs: bool = False,
    ):
        optimizer = optim.Adam([
            self.means, self.scales, self.rotations,
            self.rgbs, self.opacities
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()

        frames = []
        times = [0,0]  # rasterization, backward
        K = torch.tensor([
            [self.focal, 0, self.W/2],
            [0, self.focal, self.H/2],
            [0, 0, 1],
        ], device=self.device)

        for it in range(iterations):
            start = time.time()
            optimizer.zero_grad()

            means_3d = torch.cat([self.means, self.z_means], dim=1)   # [N,3]
            scales_3d = torch.cat([self.scales, self.scales_z], dim=1)
            quats = torch.stack([
                torch.cos(self.rotations/2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations/2)
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

            out_img = renders[0]  # [H,W,3], range ~ [0..1]
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # === マスクかける: 背景は(0,0,0) ===
            #   self.gt_image も既に α乗算で背景黒なので、合わせる。
            mask3 = self.alpha_mask.expand(-1, -1, 3)  # [H,W,3]
            out_img_masked = out_img * mask3
            gt_masked = self.gt_image * mask3

            loss = mse_loss_fn(out_img_masked, gt_masked)

            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            optimizer.step()
            self.losses.append(loss.item())
            self.meta = meta

            if (it+1) % 100 == 0:
                print(f"[Iter {it+1}/{iterations}] Loss={loss.item():.6f}")

            if save_imgs and it % 5 == 0:
                frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                frames.append(frame)

        if save_imgs and len(frames)>0:
            out_dir = os.path.join(os.getcwd(),"renders")
            os.makedirs(out_dir, exist_ok=True)
            from PIL import Image
            frames_pil = [Image.fromarray(f) for f in frames]
            frames_pil[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames_pil[1:],
                duration=5,
                loop=0,
            )

        print(f"[Timing] Rasterize={times[0]:.3f}s, Backward={times[1]:.3f}s (total)")
        original_gs, projected_gs = self.get_gaussians()
        self.save_gaussians(original_gs, projected_gs, 'fitted_gaussians.pkl')
        return original_gs, projected_gs

    def plot_loss(self, save_path="loss_curve.png"):
        import matplotlib.pyplot as plt
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


def main():
    parser = argparse.ArgumentParser(description="Train 2D Gaussians with alpha-masked background.")
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
