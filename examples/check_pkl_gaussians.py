import os
import pickle
from typing import Optional
import math
import torch
import numpy as np
from PIL import Image
from gsplat import rasterization
from twodgs import TwoDGaussians
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def load_gaussians(pickle_path: str, device: torch.device) -> tuple:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        original_gaussians = data["original_gaussians"]
        projected_gaussians = data["projected_gaussians"]
        viewmat = torch.tensor(data["viewmat"], dtype=torch.float32, device=device)
        K = torch.tensor(data["K"], dtype=torch.float32, device=device)
    return original_gaussians, projected_gaussians, viewmat, K

def draw_ellipse(ax, mean, cov, n_std=2.0, edgecolor='red'):
    """
    ガウス分布を表す楕円を描画する。

    Args:
        ax: matplotlibのAxesオブジェクト。
        mean: ガウス分布の平均値 (x, y)。
        cov: ガウス分布の共分散行列 [[xx, xy], [yx, yy]]。
        n_std: 楕円のサイズを決める標準偏差の数。
        edgecolor: 楕円の縁の色。
    """
    # 共分散行列から楕円のパラメータを取得
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    # 楕円の角度（ラジアンから度に変換）
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))

    # 楕円の半径（標準偏差のn倍）
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # 楕円を作成
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=edgecolor, facecolor='none', linewidth=1)

    # 楕円を描画
    ax.add_patch(ellipse)

def render_gaussians(
    original_gaussians: TwoDGaussians,
    projected_gaussians: TwoDGaussians,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    image_size: tuple = (256, 256),
    draw_ellipses: bool = True,  # 楕円を描画するかどうかのフラグ
) -> np.ndarray:
    device = viewmat.device  # viewmat のデバイスを取得

    # original_gaussians を使用してレンダリングを行う
    means = torch.tensor(original_gaussians.means, device=device, dtype=torch.float32)        # [k, 2]
    rotations = torch.tensor(original_gaussians.rotations, device=device, dtype=torch.float32)  # [k]
    scales = torch.tensor(original_gaussians.scales, device=device, dtype=torch.float32)        # [k, 2]
    rgbs = torch.tensor(original_gaussians.rgb, device=device, dtype=torch.float32)            # [k, 3]
    alphas = torch.tensor(original_gaussians.alpha, device=device, dtype=torch.float32)        # [k]

    # 3DのMeanを再構築（z=0を追加）
    z_means = torch.zeros((means.shape[0], 1), device=device)  # [k, 1]
    means_3d = torch.cat([means, z_means], dim=1)  # [k, 3]

    # Z軸のスケールは1に固定
    scales_z = torch.ones((scales.shape[0], 1), device=device)  # [k, 1]
    scales_3d = torch.cat([scales, scales_z], dim=1)  # [k, 3]

    # クォータニオンを再構築（Z軸周りの回転）
    quats = torch.stack([
        torch.cos(rotations / 2),
        torch.zeros_like(rotations),
        torch.zeros_like(rotations),
        torch.sin(rotations / 2)
    ], dim=1)  # [k, 4]
    quats_normalized = quats / quats.norm(dim=-1, keepdim=True)  # [k, 4]

    # ラスタライゼーションを実行
    renders, _, _ = rasterization(
        means_3d,                # [k, 3]
        quats_normalized,        # [k, 4]
        scales_3d,               # [k, 3]
        alphas,                  # [k]
        rgbs,                    # [k, 3]
        viewmat[None],           # [1, 4, 4]
        K[None],                 # [1, 3, 3]
        image_size[0],
        image_size[1],
        packed=False  # 追加：packed=Falseに設定
    )
    out_img = renders[0].detach().cpu().numpy()  # [H, W, 3]

    # uint8に変換して0-255の範囲に収める
    out_img_uint8 = (out_img * 255).clip(0, 255).astype(np.uint8)

    # 楕円を描画する場合
    if draw_ellipses and projected_gaussians is not None:
        # ガウス分布の2D位置と共分散行列を取得
        means2d = projected_gaussians.means  # [N, 2]
        covs2d = projected_gaussians.covs    # [N, 2, 2]

        # 画像に楕円を重ねて描画
        fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
        ax.imshow(out_img_uint8)
        for mean, cov in zip(means2d, covs2d):
            draw_ellipse(ax, mean, cov)

        ax.set_xlim(0, image_size[0])
        ax.set_ylim(image_size[1], 0)  # Y軸を反転
        ax.axis('off')

        # 画像を保存するためにFigureをCanvasから取得
        fig.canvas.draw()
        rendered_with_ellipses = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rendered_with_ellipses = rendered_with_ellipses.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))

        plt.close(fig)

        return rendered_with_ellipses  # 楕円を描画した画像を返す

    return out_img_uint8

def save_image(image_array: np.ndarray, save_path: str):
    img = Image.fromarray(image_array)
    img.save(save_path)
    print(f"Rendered image saved to {save_path}")

def main(pickle_path: str, output_image_path: str, image_size: tuple = (256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_gaussians, projected_gaussians, viewmat, K = load_gaussians(pickle_path, device=device)
    rendered_image = render_gaussians(original_gaussians, projected_gaussians, viewmat, K, image_size=image_size, draw_ellipses=True)
    save_image(rendered_image, output_image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Gaussians from Pickle File")
    parser.add_argument("--pickle_path", type=str, help="Path to the fitted_gaussians.pkl file")
    parser.add_argument("--output_image_path", type=str, help="Path to save the rendered image")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1554, 1162], help="Width and height of the output image")

    args = parser.parse_args()
    main(args.pickle_path, args.output_image_path, tuple(args.image_size))
