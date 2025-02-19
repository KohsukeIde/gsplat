#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import time
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
import torch

from gsplat import rasterization
from twodgs import TwoDGaussians
from pytorch_msssim import ssim as ssim_fn
from PIL import Image


def calculate_psnr(img: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """Calculate PSNR between img and gt."""
    mse = np.mean((img - gt) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def calculate_ssim(img: np.ndarray, gt: np.ndarray, device: torch.device = torch.device('cpu')) -> float:
    """
    Calculate SSIM between img and gt in [0..1], shape [H,W,3].
    pytorch_msssimを使用。
    """
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()
    gt_t  = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        ssim_val = ssim_fn(img_t, gt_t, data_range=1.0, size_average=True)
    return ssim_val.item()

def rgba_path_to_tensor(image_path: Path, device: torch.device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    4チャンネル RGBA 画像を読み込み、(rgb, alpha) の2つに分けて返す。
      - rgb:   [H,W,3] in [0..1]
      - alpha: [H,W,1] in [0..1], 0=背景, 1=前景など
    """
    from torchvision import transforms
    img = Image.open(image_path).convert("RGBA")  # 4ch
    t = transforms.ToTensor()(img)  # shape [4,H,W], in [0..1]
    t = t.permute(1,2,0)            # [H,W,4]
    rgb   = t[..., :3]             # [H,W,3]
    alpha = t[...,  3:]            # [H,W,1]

    if device is not None:
        rgb   = rgb.to(device)
        alpha = alpha.to(device)

    if alpha == None:
        print("where is alpha")
        sys.exit()
    return rgb, alpha

def load_images_from_folder(folder: Path) -> List[Path]:
    """指定フォルダ内の画像パス一覧をソートして返す。"""
    valid_exts = ('.jpg','.jpeg','.png','.bmp','.tiff')
    image_paths = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in valid_exts:
            image_paths.append(f)
    image_paths.sort()
    return image_paths


class SimpleTrainer:
    """
    2Dガウシアンの集合を使ってRGBA画像を再現するように学習するクラス。
    """

    def __init__(
        self,
        gt_image: torch.Tensor,       # [H,W,3], 0..1
        alpha_mask: Optional[torch.Tensor] = None,  # [H,W,1], 0..1 or None
        num_points: int = 32,
        device: torch.device = torch.device('cpu'),
        initial_params: dict = None,  # ここに { "means": Tensor, ... } を渡すと初期値を上書き
    ):
        """
        gt_image: 学習先のRGBテンソル ([H,W,3])
        alpha_mask: 前景マスク ([H,W,1])。なければ None。
        num_points: ガウシアン数
        device: 'cpu' or 'cuda:0' など
        initial_params: 前の学習結果を受け継ぎたい場合は dict を渡す
        """
        self.device = device
        self.gt_image   = gt_image
        self.alpha_mask = alpha_mask
        self.num_points = num_points

        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        # fov_x = 90度相当 (orthographicではあまり重要でない)
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)

        # 学習ログ (MSE loss) を記録する
        self.losses = []
        # rasterization メタ情報
        self.meta = None

        # ガウシアンパラメータを初期化
        self._init_gaussians(alpha_mask, initial_params)
        print(f"[Trainer] H={self.H}, W={self.W}, num_points={self.num_points}, device={self.device}")

    def _init_gaussians(
        self,
        alpha_mask: Optional[torch.Tensor],
        initial_params: dict
    ):
        """
        ガウシアン中心・スケール・色などを初期化。
        initial_params が指定されていれば、それを使って初期化。
        """
        bd = 2
        d  = 3

        if initial_params is not None:
            print("[Trainer] Using previous fitting results as initial parameters.")
            self.means     = initial_params["means"].clone().detach().to(self.device)
            self.scales    = initial_params["scales"].clone().detach().to(self.device)
            self.rotations = initial_params["rotations"].clone().detach().to(self.device)
            self.rgbs      = initial_params["rgbs"].clone().detach().to(self.device)
            self.opacities = initial_params["opacities"].clone().detach().to(self.device)
        else:
            # ランダム初期化
            if alpha_mask is None:
                # 全画面にランダム配置: [-1, +1]
                self.means = bd * (torch.rand(self.num_points, 2, device=self.device) - 0.5)
            else:
                # 前景マスクの画素からサンプリング
                mask_2d = alpha_mask[..., 0]
                foreground_indices = torch.nonzero(mask_2d > 0.5, as_tuple=False)
                K = foreground_indices.shape[0]
                if K < self.num_points:
                    print("Warning: foregroundがガウシアン数より少ない。重複サンプリングします。")
                chosen = torch.randint(0, K, size=(self.num_points,), device=self.device)
                chosen_xy = foreground_indices[chosen]  # [N,2] (y,x)
                y = chosen_xy[:, 0].float()
                x = chosen_xy[:, 1].float()
                x_norm = (x / (self.W - 1)) * 2.0 - 1.0
                y_norm = (y / (self.H - 1)) * 2.0 - 1.0
                self.means = torch.stack([x_norm, y_norm], dim=-1)

            self.scales    = torch.rand(self.num_points, 2, device=self.device)
            self.rotations = torch.rand(self.num_points, device=self.device) * 2 * math.pi
            self.rgbs      = torch.rand(self.num_points, d, device=self.device)
            self.opacities = torch.ones(self.num_points, device=self.device)

        # z 関連は固定
        self.z_means  = torch.ones(self.num_points, 1, device=self.device)
        self.scales_z = torch.ones(self.num_points, 1, device=self.device)

        # ビュー行列 (orthographic)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        # 学習対象
        self.means.requires_grad     = True
        self.scales.requires_grad    = True
        self.rotations.requires_grad = True
        self.rgbs.requires_grad      = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad   = False

    def _get_covs(self) -> torch.Tensor:
        """(scales, rotations) から 2x2 の共分散行列を計算。"""
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
        original_gaussians, projected_gaussians のタプルを返す。
        projected_gaussians は rasterization のメタ情報があれば計算。
        """
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
            means2d = self.meta['means2d'].detach().cpu().numpy()  # [M,2]
            conics  = self.meta['conics'].detach().cpu().numpy()   # [M,3]
            # conic(A,B,C) -> 2x2共分散へ
            A = conics[:,0]
            B = conics[:,1]
            C = conics[:,2]
            M = means2d.shape[0]
            inv_covs = np.zeros((M,2,2))
            inv_covs[:,0,0] = A
            inv_covs[:,0,1] = B/2
            inv_covs[:,1,0] = B/2
            inv_covs[:,1,1] = C
            covs2d = np.linalg.inv(inv_covs)

            # RGBやalphaは単純に先頭M個を切り出す想定 (もしくは全Nを使うなら要注意)
            rgb_ = torch.sigmoid(self.rgbs).detach().cpu().numpy()[:M]
            alpha_ = torch.sigmoid(self.opacities).detach().cpu().numpy()[:M]

            projected_gaussians = TwoDGaussians(
                means=means2d,
                covs=covs2d,
                rgb=rgb_,
                alpha=alpha_,
                rotations=self.rotations.detach().cpu().numpy()[:M],
                scales=self.scales.detach().cpu().numpy()[:M],
            )
        else:
            projected_gaussians = None

        return original_gaussians, projected_gaussians

    def save_gaussians(
        self,
        original_gaussians: TwoDGaussians,
        projected_gaussians: Optional[TwoDGaussians],
        file_path: str
    ):
        """
        original_gaussians, projected_gaussians, カメラ行列などを pickle 化して保存
        """
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
        print(f"Saved Gaussians to {file_path}")

    def train(
        self,
        max_iterations: int = 1000,
        lr: float = 0.01,
        capture_step: int = 100,
    ) -> None:
        """
        学習ループを回す。
          - max_iterations: 総イテレーション回数
          - lr: Learning Rate
          - capture_step: フレームを保存するステップ間隔 (例: 100なら100iterごとに保存)
        """
        optimizer = torch.optim.Adam([
            self.means, self.scales, self.rotations, self.rgbs, self.opacities
        ], lr=lr)
        mse_loss_fn = torch.nn.MSELoss()

        # カメラ内パラメータ
        K = torch.tensor([
            [self.focal, 0, self.W / 2],
            [0, self.focal, self.H / 2],
            [0, 0, 1],
        ], device=self.device)

        # フレーム保存用
        self.saved_frames = []

        for i in range(max_iterations):
            optimizer.zero_grad()
            # 3D 座標
            means_3d = torch.cat([self.means, self.z_means], dim=1)   # [N,3]
            scales_3d= torch.cat([self.scales, self.scales_z], dim=1)
            # 2D 回転を簡易クォータニオンに
            quats = torch.stack([
                torch.cos(self.rotations / 2),
                torch.zeros_like(self.rotations),
                torch.zeros_like(self.rotations),
                torch.sin(self.rotations / 2)
            ], dim=1)
            quats_norm = quats / quats.norm(dim=-1, keepdim=True)

            # RGBA (r,g,b,a) in [0..1]
            rgba_for_raster = torch.cat([
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities).unsqueeze(-1)
            ], dim=-1)  # [N,4]

            dummy_opacities = torch.ones_like(self.opacities)

            # ラスタライズ
            renders, _, meta = rasterization(
                means_3d,
                quats_norm,
                scales_3d,
                dummy_opacities,
                rgba_for_raster,
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
            )

            out_rgba = renders[0]  # shape [H,W,4]
            out_rgb   = out_rgba[..., :3]
            out_alpha = out_rgba[..., 3:4]
            out_pre   = out_rgb * out_alpha  # 背景黒で合成

            # GT も alpha_mask があれば掛け算
            if self.alpha_mask is not None:
                gt_pre = self.gt_image * self.alpha_mask.expand(-1, -1, 3)
            else:
                gt_pre = self.gt_image

            loss = mse_loss_fn(out_pre, gt_pre)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            self.meta = meta

            # capture_step 毎にフレームを保存 & ログ
            if (i+1) % capture_step == 0 or (i == max_iterations - 1):
                frame_rgb = (out_pre.detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
                self.saved_frames.append(frame_rgb)

                # 進捗表示 (PSNR, SSIM)
                final_out_img = out_pre.detach().cpu().numpy()  # [H,W,3]
                gt_np = gt_pre.detach().cpu().numpy()
                psnr_val  = calculate_psnr(final_out_img, gt_np, max_val=1.0)
                ssim_val  = calculate_ssim(final_out_img, gt_np, device=self.device)
                print(f"Iteration {i+1}/{max_iterations}, Loss={loss.item():.6f}, "
                      f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")

    def get_final_params(self) -> dict:
        """
        学習後のパラメータを dict で返し、次の画像の初期値として引き継げる。
        """
        return {
            "means":     self.means.detach().clone(),
            "scales":    self.scales.detach().clone(),
            "rotations": self.rotations.detach().clone(),
            "rgbs":      self.rgbs.detach().clone(),
            "opacities": self.opacities.detach().clone(),
        }


def main():
    #=== ユーザ設定 ===#
    folder = Path("/groups/gag51404/ide/data/DTU/scan63/images")
    
    # 出力フォルダを整理
    output_base = Path("sequential_fitting_output")
    output_base.mkdir(exist_ok=True)
    
    gif_dir = output_base / "gifs"
    pkl_dir = output_base / "pkls"
    gif_dir.mkdir(exist_ok=True)
    pkl_dir.mkdir(exist_ok=True)

    num_points    = 500
    max_iters     = 10000
    lr            = 0.01
    capture_step  = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = load_images_from_folder(folder)
    if not image_paths:
        print(f"No valid images found in {folder}")
        return

    print("Image list:")
    for p in image_paths:
        print(" -", p.name)

    # 連番画像を順番に処理しつつ，学習結果(ガウシアン)を引き継ぐ
    prev_params = None  # 最初は None

    for idx, img_path in enumerate(image_paths):
        print(f"\n===== Fitting to {img_path.name} (index {idx}) =====")
        gt_rgb, alpha_mask = rgba_path_to_tensor(img_path, device=device)

        # Trainer を用意 (前回の結果があれば initial_params に渡す)
        trainer = SimpleTrainer(
            gt_image=gt_rgb,
            alpha_mask=alpha_mask,
            num_points=num_points,
            device=device,
            initial_params=prev_params
        )

        # 学習 (指定ステップごとにフレームをキャプチャ)
        trainer.train(max_iterations=max_iters, lr=lr, capture_step=capture_step)

        # === GIF 保存 ===
        frames = trainer.saved_frames
        out_gif_name = gif_dir / f"train_{idx:04d}.gif"
        if len(frames) > 0:
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                out_gif_name,
                save_all=True,
                append_images=pil_frames[1:],
                duration=30,
                loop=0
            )
            print(f"Saved GIF for {img_path.name} => {out_gif_name}")
        else:
            print("No frames captured, no GIF output.")
        # フレームを破棄（メモリ解放）
        trainer.saved_frames.clear()

        # === PKL 保存 ===
        original_gs, projected_gs = trainer.get_gaussians()
        out_pkl_name = pkl_dir / f"fitted_gaussians_{idx:04d}.pkl"
        trainer.save_gaussians(original_gs, projected_gs, str(out_pkl_name))

        # 次の画像にパラメータを引き継ぐ
        prev_params = trainer.get_final_params()

    print("All done!")


if __name__ == "__main__":
    main()
