"""
This finds and collects a single, representative example for each VLM.

For each model specified in `MODELS_TO_PROCESS`, the script searches within the
`syn_nio` subfolder to find the first text file and its corresponding PNG image.
It then copies both the image and the text file to a centralized `OUTPUT_DIR`,
renaming them with the model's name as a prefix to avoid naming conflicts.

This is useful for quickly gathering a sample of generated NIO synthetic
images and their captions from each model for qualitative review or for inclusion
in a report or presentation.
"""

import os
import shutil

# config
BASE_DATASETS_DIR = "../../data"
MODELS_TO_PROCESS = ["BLIP", "Florence2", "BLIP-LoRA"]
SUBFOLDER_TO_SEARCH = "syn_nio"
OUTPUT_DIR = "../../outputs/caption_examples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for model_name in MODELS_TO_PROCESS:
    print(f"Processing model: {model_name}")

    try:
        example_dir = os.path.join(BASE_DATASETS_DIR, model_name, SUBFOLDER_TO_SEARCH)

        if not os.path.isdir(example_dir):
            print(f"Directory not found: {example_dir}")
            continue

        found_txt_file = None
        for filename in sorted(os.listdir(example_dir)):
            if filename.lower().endswith(".txt"):
                found_txt_file = filename
                break

        if not found_txt_file:
            print(f"No text files found in {example_dir}")
            continue

        base_name = os.path.splitext(found_txt_file)[0]
        image_filename = f"{base_name}.png"

        source_txt_path = os.path.join(example_dir, found_txt_file)
        source_image_path = os.path.join(example_dir, image_filename)

        if not os.path.exists(source_image_path):
            print(f"Found caption '{found_txt_file}' but missing image '{image_filename}'")
            continue

        with open(source_txt_path, 'r') as f:
            caption_content = f.read().strip()

        print(f"Found example: {base_name}")
        print(f"Caption: \"{caption_content}\"")

        dest_txt_path = os.path.join(OUTPUT_DIR, f"{model_name}_{found_txt_file}")
        dest_image_path = os.path.join(OUTPUT_DIR, f"{model_name}_{image_filename}")

        shutil.copyfile(source_txt_path, dest_txt_path)
        shutil.copyfile(source_image_path, dest_image_path)
        print(f"Copied files to '{OUTPUT_DIR}'")

    except Exception as e:
        print(f"Error occurred: {e}")
