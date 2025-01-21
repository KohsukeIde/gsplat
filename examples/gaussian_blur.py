import cv2
import os
import argparse
from pathlib import Path

def apply_gaussian_filter(input_folder: Path, output_folder: Path):
    """
    Applies a Gaussian filter to all images in 'input_folder' and saves
    the results in 'output_folder' with filenames like 'image_1.png', etc.
    """

    # Create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect all valid image paths
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = [f for f in input_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in valid_exts]
    image_paths.sort()

    for i, img_path in enumerate(image_paths, start=1):
        # Read the image in color (BGR) mode
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping file (not an image or unreadable): {img_path}")
            continue
        
        # Apply a Gaussian blur with a 5x5 kernel (adjust as needed)
        # The second param is the kernel size (width, height),
        # The third param is standard deviation in X direction (0 => auto).
        gaussian_img = cv2.GaussianBlur(img, (77, 77), 0)
        
        # Construct the output filename: image_1.png, image_2.png, etc.
        out_filename = f"image_{i}.png"  
        out_path = output_folder / out_filename
        
        # Save the resulting image
        cv2.imwrite(str(out_path), gaussian_img)
        print(f"Saved blurred image to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Apply a Gaussian filter to all images in a folder."
    )
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the input folder containing images')
    parser.add_argument('--output_folder', type=str, default='gaussian_filter_image',
                        help='Folder name or path to save blurred images')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    # Run the function
    apply_gaussian_filter(input_folder, output_folder)

if __name__ == "__main__":
    main()
