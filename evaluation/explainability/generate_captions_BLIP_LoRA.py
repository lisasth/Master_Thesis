# this script is only runnable with GPU support and not on Apple Silicon because of bitsandbytes
"""
This generates image captions using a base BLIP model that has been fine-tuned with LoRA.
It is specifically configured to leverage a smal base model ('Salesforce/blip-image-captioning-base')
and is optimized for a CUDA-enabled GPU environment.

The script performs the following steps:
1.  Loads the base BLIP processor and model.
2.  Injects the trained LoRA adapter weights from a specified checkpoint directory
    into the base model to create the fine-tuned version.
3.  Iterates through a predefined list of input image directories (DOMAINS).
4.  For each image found, it generates a descriptive caption using the
    fine-tuned model.
5.  Saves each generated caption as a separate .txt file in a structured
    output directory, organized by the original domain of the image.
"""

import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from pathlib import Path

# config
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DOMAINS = ["real_io", "real_nio", "IO", "NIO"]
LORA_ADAPTER_DIR = "../../models/BLIP_LoRA/checkpoint-epoch30/"
OUTPUT_ROOT = Path("outputs/BLIP-LoRA")
MAX_LENGTH = 50


def load_lora_model_and_processor(adapter_dir: str):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    base_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_dir).to(DEVICE)
    model_with_lora.eval()
    return processor, model_with_lora


def generate_caption_for_image(processor, model, image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {image_path}: {e}")
        return ""

    # tokenize using BLIP2
    inputs = processor(
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=3,
            no_repeat_ngram_size=2
        )
    caption = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return caption


def main():
    print("Loading BLIP‐2 + LoRA …")
    processor, model = load_lora_model_and_processor(LORA_ADAPTER_DIR)

    for domain in DOMAINS:
        input_folder = Path("datasets/BLIP") / domain
        if not input_folder.exists():
            print(f"Skipping missing folder: {input_folder}")
            continue

        out_folder = OUTPUT_ROOT / domain
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f"\nDomain: {domain}, generating captions into {out_folder}/ ...")
        for fname in sorted(os.listdir(input_folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = str(input_folder / fname)
            caption = generate_caption_for_image(processor, model, image_path)

            txt_name = fname.rsplit(".", 1)[0] + ".txt"
            with open(out_folder / txt_name, "w") as f_txt:
                f_txt.write(caption)

            print(f"[LoRA][{domain}] {fname} → {caption}")


if __name__ == "__main__":
    main()
