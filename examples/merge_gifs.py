from PIL import Image, ImageSequence
import glob
import os
from tqdm import tqdm

gif_folder = "/groups/gag51404/ide/gsplat/examples/63_8_2K_gifs"
gif_files = sorted(glob.glob(os.path.join(gif_folder, "*.gif")))

frames = []
for gif_file in tqdm(gif_files, desc="Reading multi-frame GIFs"):
    im = Image.open(gif_file)
    for frame in ImageSequence.Iterator(im):
        # フレームのモードに応じて変換が必要な場合もある
        frames.append(frame.convert('RGBA'))

frames[0].save(
    "merged_multiframe.gif",
    save_all=True,
    append_images=frames[1:],
    duration=200,
    loop=0
)
print("結合完了!")
