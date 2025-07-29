"""
This script provides a utility to recursively find all images within a specified
input directory, resize them to a defined target size, and save the results
to an output directory while preserving the original folder structure.
"""

import os
from PIL import Image

input = "/Users/stuch/Desktop/MA_2025/Images/Donuts_img/CLIP_30_70"
output = "/Users/stuch/Desktop/CLIP_30_70_resized"
size = (256, 256)


def resize(input_dir, output_dir, s):
    os.makedirs(output, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                in_path = os.path.join(root, file)
                out_path = os.path.join(save_path, file)

                try:
                    with Image.open(in_path) as img:
                        img_resize = img.resize(s, Image.BICUBIC)
                        img_resize.save(out_path)
                except Exception as e:
                    print("error")


resize(input, output, size)
