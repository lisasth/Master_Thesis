"""
This generates and compares image captions using the BLIP model.

1.  Caption Generation:
    - It recursively discovers all image datasets located under a specified root
      directory.
    - For each image found, it uses the pre-trained 'Salesforce/blip-image-captioning-large'
      model to generate a descriptive caption.
    - Each generated caption is saved as a '.txt' file alongside its corresponding
      image, and all captions are also compiled into a master CSV file for easy access.

2.  Caption Similarity Comparison:
    - It identifies pairs of corresponding real and synthetic images (based on filename).
    - For each pair, it calculates the semantic similarity between their generated
      captions using a SentenceTransformer ('all-MiniLM-L6-v2').
    - The results of this comparison, including the real caption, synthetic caption,
      and their cosine similarity score, are saved to a separate CSV file.

The final outputs are two CSV files located in the 'outputs/' directory:
- `captions_BLIP.csv`: A comprehensive log of all generated captions.
- `caption_similarity_BLIP.csv`: A detailed comparison of real vs. synthetic captions.
"""

import csv
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from typing import Union

# config
root_dir = Path("../../data/single")
captions_output_csv = "../../outputs/reports/captions_BLIP.csv"
comparison_output_csv = "../../outputs/reports/caption_similarity_BLIP.csv"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# sentence transformer for comparison
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def generate_caption(image_path: Path) -> Union[str, None]:
    """Generates a caption for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_length=50, num_beams=3)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None


def compare_captions(cap1: str, cap2: str) -> Union[float, None]:
    """Calculates cosine similarity between two caption strings."""
    if cap1 is None or cap2 is None:
        return None
    emb1 = embedder.encode(cap1, convert_to_tensor=True)
    emb2 = embedder.encode(cap2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# main logic
# 1. discover all images and generate captions
all_captions = {}
all_results_for_csv = []

# find all 'IO' and 'NIO' folders within the directory structure
# structure: Product_GenModel/DataType/ImageType, e.g., Croissants_DM/real/IO
image_folders = list(root_dir.glob("*/*/*"))

print(f"\nFound {len(image_folders)} image folders to process.")

for folder_path in image_folders:
    try:
        image_type = folder_path.name  # IO or NIO
        data_type = folder_path.parent.name  # real or syn
        product_gen_model = folder_path.parent.parent.name  # e.g., Croissants_DM
        product, gen_model = product_gen_model.rsplit('_', 1)

    except (ValueError, IndexError):
        print(f"Skipping malformed directory: {folder_path}")
        continue

    print(f"\nProcessing: product='{product}', model='{gen_model}', type='{data_type}/{image_type}'")

    all_captions.setdefault(product, {}).setdefault(gen_model, {}).setdefault(data_type, {}).setdefault(image_type, {})

    for fpath in sorted(folder_path.glob("*.png")):
        fname = fpath.name
        caption = generate_caption(fpath)

        if caption:
            print(f"  - {fname} → {caption}")
            all_captions[product][gen_model][data_type][image_type][fname] = caption

            # save caption to a .txt file next to the image
            txt_path = fpath.with_suffix(".txt")
            with open(txt_path, "w") as txt_file:
                txt_file.write(caption)

            all_results_for_csv.append({
                "product": product,
                "gen_model": gen_model,
                "data_type": data_type,
                "image_type": image_type,
                "filename": fname,
                "caption": caption
            })

# save all generated captions to a CSV file
print(f"\nSaving all captions to '{captions_output_csv}'")
with open(captions_output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["product", "gen_model", "data_type", "image_type", "filename", "caption"])
    writer.writeheader()
    writer.writerows(all_results_for_csv)

# 2. compare captions between real and synthetic pairs
print("\nStarting Caption Comparison")
comparison_results = []

for product, gen_models in all_captions.items():
    for gen_model, data_types in gen_models.items():
        if 'real' not in data_types or 'syn' not in data_types:
            continue

        print(f"\nComparing '{product}' captions from generative model '{gen_model}'")

        # compare IO with IO and NIO with NIO
        for image_type in ['IO', 'NIO']:
            if image_type not in data_types['real'] or image_type not in data_types['syn']:
                continue

            real_images = data_types['real'][image_type]
            syn_images = data_types['syn'][image_type]
            common_filenames = set(real_images.keys()) & set(syn_images.keys())

            if not common_filenames:
                print(f"No common images found for type '{image_type}'")
                continue

            print(f"Comparing {len(common_filenames)} images of type '{image_type}'")
            for fname in sorted(list(common_filenames)):
                cap_real = real_images[fname]
                cap_syn = syn_images[fname]
                score = compare_captions(cap_real, cap_syn)

                print(f"{fname}: Similarity = {score:.4f}")

                comparison_results.append({
                    "product": product,
                    "gen_model": gen_model,
                    "image_type": image_type,
                    "image_filename": fname,
                    "real_caption": cap_real,
                    "syn_caption": cap_syn,
                    "similarity_score": score
                })

# 3. Save comparison results to a final CSV
print(f"\nSaving comparison results to '{comparison_output_csv}'...")
with open(comparison_output_csv, "w", newline="") as f:
    fieldnames = [
        "product", "gen_model", "image_type", "image_filename",
        "real_caption", "syn_caption", "similarity_score"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_results)
