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
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
            # Debug output to see all available data
            print(f"PKL file keys: {list(data.keys())}")
            
            # Extract data from the pickle file
            original_gaussians = data["original_gaussians"]
            projected_gaussians = data.get("projected_gaussians", None)
            viewmat = torch.tensor(data["viewmat"], dtype=torch.float32, device=device)
            K = torch.tensor(data["K"], dtype=torch.float32, device=device)
            
            # Extract image size if available, otherwise None
            image_size = data.get("image_size", None)
            if image_size:
                print(f"Found image size in PKL: {image_size}")
            else:
                print("No image_size in PKL, will estimate from K matrix")
                
        return original_gaussians, projected_gaussians, viewmat, K, image_size
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        print(f"File path: {pickle_path}")
        print(f"File exists: {os.path.exists(pickle_path)}")
        print(f"File size: {os.path.getsize(pickle_path) if os.path.exists(pickle_path) else 'N/A'} bytes")
        raise

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
    image_size: tuple = None,
    draw_ellipses: bool = True,
) -> np.ndarray:
    device = viewmat.device
    
    # Fallback to estimating from K if image_size is None
    if image_size is None:
        width = int(K[0, 2].item() * 2)
        height = int(K[1, 2].item() * 2)
        image_size = (width, height)
        print(f"Using estimated image size from K: {image_size}")
    
    print(f"Rendering with dimensions: {image_size}")
    
    # Debug information to understand what's in the PKL file
    print(f"Original gaussians: {len(original_gaussians.means)} points")
    if projected_gaussians:
        print(f"Projected gaussians: {len(projected_gaussians.means)} points")
    else:
        print("No projected gaussians")
    
    # Get original_gaussians parameters
    means = torch.tensor(original_gaussians.means, device=device, dtype=torch.float32)
    rotations = torch.tensor(original_gaussians.rotations, device=device, dtype=torch.float32)
    scales = torch.tensor(original_gaussians.scales, device=device, dtype=torch.float32)
    rgbs = torch.tensor(original_gaussians.rgb, device=device, dtype=torch.float32)
    alphas = torch.tensor(original_gaussians.alpha, device=device, dtype=torch.float32)
    
    # Build 3D parameters
    z_means = torch.zeros((means.shape[0], 1), device=device)
    means_3d = torch.cat([means, z_means], dim=1)
    
    scales_z = torch.ones((scales.shape[0], 1), device=device)
    scales_3d = torch.cat([scales, scales_z], dim=1)
    
    # Create quaternions
    quats = torch.stack([
        torch.cos(rotations / 2),
        torch.zeros_like(rotations),
        torch.zeros_like(rotations),
        torch.sin(rotations / 2)
    ], dim=1)
    quats_normalized = quats / quats.norm(dim=-1, keepdim=True)
    
    # Combine RGB and Alpha into RGBA to match training
    rgba_for_raster = torch.cat([rgbs, alphas.unsqueeze(-1)], dim=-1)
    
    # Use dummy_opacities like in training
    dummy_opacities = torch.ones_like(alphas)
    
    # Call rasterization with camera_model="ortho" to match training
    renders, _, _ = rasterization(
        means_3d,
        quats_normalized,
        scales_3d,
        dummy_opacities,
        rgba_for_raster,
        viewmat[None],
        K[None],
        image_size[0],
        image_size[1],
        camera_model="ortho",  # Important: Match the training setting
        packed=False
    )
    
    # Process output like in training
    out_rgba = renders[0].detach().cpu().numpy()
    
    # Apply pre-multiplied alpha like in training
    out_rgb = out_rgba[..., :3]
    out_alpha = out_rgba[..., 3:4]
    out_premult = out_rgb * out_alpha
    
    # Convert to uint8
    out_img_uint8 = (out_premult * 255).clip(0, 255).astype(np.uint8)
    
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
        rendered_with_ellipses = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        rendered_with_ellipses = rendered_with_ellipses.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 3))

        plt.close(fig)

        return rendered_with_ellipses  # 楕円を描画した画像を返す

    return out_img_uint8

def save_image(image_array: np.ndarray, save_path: str):
    img = Image.fromarray(image_array)
    img.save(save_path)
    print(f"Rendered image saved to {save_path}")

def main(pickle_path: str, output_dir: str = None, image_size: tuple = None):
    """
    Process a single pickle file or all pickle files in a directory.
    
    Args:
        pickle_path: Path to a pickle file or directory containing pickle files
        output_dir: Directory to save output images (defaults to same as input)
        image_size: Size of the output image (only used if not in the PKL)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if pickle_path is a directory
    if os.path.isdir(pickle_path):
        # Process all pickle files in the directory
        if output_dir is None:
            output_dir = pickle_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all pickle files
        pkl_files = [f for f in os.listdir(pickle_path) if f.endswith('.pkl')]
        
        if not pkl_files:
            print(f"No pickle files found in {pickle_path}")
            return
            
        print(f"Found {len(pkl_files)} pickle files to process")
        
        # Process each pickle file
        for pkl_file in pkl_files:
            pkl_path = os.path.join(pickle_path, pkl_file)
            # Create output filenames with same base name but .png extension
            base_filename = os.path.splitext(pkl_file)[0]
            output_path_without_ellipses = os.path.join(output_dir, base_filename + '_without_ellipses.png')
            output_path_with_ellipses = os.path.join(output_dir, base_filename + '_with_ellipses.png')
            
            try:
                print(f"Processing {pkl_file}...")
                original_gaussians, projected_gaussians, viewmat, K, pkl_image_size = load_gaussians(pkl_path, device=device)
                
                # Use the image size from PKL if available, otherwise use the provided default
                current_image_size = pkl_image_size if pkl_image_size is not None else image_size
                print(f"Rendering with image size: {current_image_size}")
                
                # Render without ellipses
                rendered_image = render_gaussians(original_gaussians, projected_gaussians, viewmat, K, 
                                                 image_size=current_image_size, draw_ellipses=False)
                save_image(rendered_image, output_path_without_ellipses)
                
                # Render with ellipses
                rendered_image_with_ellipses = render_gaussians(original_gaussians, projected_gaussians, viewmat, K, 
                                                               image_size=current_image_size, draw_ellipses=True)
                save_image(rendered_image_with_ellipses, output_path_with_ellipses)
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
    else:
        # Process a single file
        if output_dir is None:
            output_dir = os.path.dirname(pickle_path)
            if not output_dir:
                output_dir = '.'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filenames with same base name but .png extension
        base_filename = os.path.splitext(os.path.basename(pickle_path))[0]
        output_path_without_ellipses = os.path.join(output_dir, base_filename + '_without_ellipses.png')
        output_path_with_ellipses = os.path.join(output_dir, base_filename + '_with_ellipses.png')
        
        original_gaussians, projected_gaussians, viewmat, K, pkl_image_size = load_gaussians(pickle_path, device=device)
        
        # Use the image size from PKL if available, otherwise use the provided default
        current_image_size = pkl_image_size if pkl_image_size is not None else image_size
        print(f"Rendering with image size: {current_image_size}")
        
        # Render without ellipses
        rendered_image = render_gaussians(original_gaussians, projected_gaussians, viewmat, K, 
                                         image_size=current_image_size, draw_ellipses=False)
        save_image(rendered_image, output_path_without_ellipses)
        
        # Render with ellipses
        rendered_image_with_ellipses = render_gaussians(original_gaussians, projected_gaussians, viewmat, K, 
                                                       image_size=current_image_size, draw_ellipses=True)
        save_image(rendered_image_with_ellipses, output_path_with_ellipses)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Gaussians from Pickle File")
    parser.add_argument("--pickle_path", type=str, help="Path to the fitted_gaussians.pkl file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output images")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1554, 1162], help="Width and height of the output image")

    args = parser.parse_args()
    main(args.pickle_path, args.output_dir, tuple(args.image_size))
