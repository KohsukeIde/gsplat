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

        # スケールや回転
        self.scales = torch.rand(self.num_points, 2, device=self.device)
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

        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(gt_img_masked)
        axs[0].set_title(f" GT Masked")
        axs[1].imshow(out_img_masked)
        axs[1].set_title(f"{title_prefix} Out (Not masked)")
        im_2 = axs[2].imshow(diff_map, cmap='jet')
        axs[2].set_title(f"MSE={mse_value:.4f}")
        fig.colorbar(im_2, ax=axs[2], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        os.makedirs("comparison", exist_ok=True)
        save_path = f"comparison/{title_prefix.replace(' ', '_')}.png"
        plt.savefig(save_path)
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

        # カメラ内部行列 K
        K = torch.tensor([
            [self.focal, 0, self.W/2],
            [0, self.focal, self.H/2],
            [0, 0, 1],
        ], device=self.device)

        for it in range(iterations):
            start = time.time()
            optimizer.zero_grad()

            # 3Dパラメータ
            means_3d = torch.cat([self.means, self.z_means], dim=1)  # [N,3]
            scales_3d= torch.cat([self.scales, self.scales_z], dim=1)
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
            )

            out_rgba  = renders[0]               # [H,W,4]
            out_rgb   = out_rgba[..., :3]        # [H,W,3]
            out_alpha = out_rgba[...,  3:4]      # [H,W,1]
            torch.cuda.synchronize()
            times[0] += time.time() - start

            # (1) 「フィッティングした色 × フィッティングした透明度」: pre-multiplied alpha
            out_pre_mult = out_rgb * out_alpha   # [H,W,3]

            # (2) GT も同様に alpha_mask を掛けたもの
            gt_pre_mult  = self.gt_image * self.alpha_mask  # [H,W,3]

            # MSE
            loss = mse_loss_fn(out_pre_mult, gt_pre_mult)

            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start

            optimizer.step()
            self.losses.append(loss.item())
            self.meta = meta

            if (it+1) % 500 == 0:
                print(f"[Iter {it+1}/{iterations}] Loss={loss.item():.6f}")
                # 可視化
                # out_pre_mult  vs  gt_pre_mult
                self.visualize_masked_results(
                    gt_img_masked=gt_pre_mult, 
                    out_img_masked=out_pre_mult, 
                    title_prefix=f"Iter_{it+1}"
                )

            if save_imgs and (it % 5 == 0):
                # ここでは背景含めた out_rgb をフレーム化 (お好みで)
                frame = (out_rgb.detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                frames.append(frame)

        # GIF 保存
        if save_imgs and frames:
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames_pil = [Image.fromarray(f) for f in frames]
            frames_pil[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames_pil[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        print(f"[Timing] Rasterization={times[0]:.3f}s, Backward={times[1]:.3f}s (total)")
        original_gs, projected_gs = self.get_gaussians()
        self.save_gaussians(original_gs, projected_gs, 'fitted_gaussians.pkl')
        return original_gs, projected_gs


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
