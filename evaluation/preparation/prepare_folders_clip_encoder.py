"""
This script automates the creation of structured training and evaluation datasets.
It takes source directories of real and synthetic images, separated by class (IO and NIO),
and generates a series of datasets with varying compositions of real-to-synthetic data.

The script performs the following key steps:
1.  Initial Data Split: It first performs a global 80/20 train/evaluation
    split on all source images. This is done only once to ensure that the
    evaluation set remains consistent across all experiments and to prevent
    any data leakage from the training set.
2.  Ratio-Based Dataset Creation: It iterates through a predefined set of
    real-to-synthetic data ratios (e.g., 100% real, 70% real/30% synthetic, etc.).
3.  Folder Generation: For each ratio, it creates a pair of uniquely named
    'train' and 'eval' folders.
4.  Data Population: It populates these folders by copying the appropriate
    percentage of images from the pre-split training and evaluation sets,
    creating balanced datasets for both 'good' (IO) and 'bad' (NIO) classes.

The output is a complete, structured set of folders ready to be used for
training and evaluating a classifier under different data conditions.
"""

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split


def get_image_files(directory):
    """Returns a list of image files from a directory."""
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found, skipping: {directory}")
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def copy_files(files, source_dir, dest_dir):
    """Copies a list of files from a source to a destination directory."""
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(dest_dir, f))


def create_datasets(args):
    """Main function to create training and evaluation datasets."""

    # define the ratios and their corresponding percentages (real, synthetic)
    ratios = {
        '100real': (1.0, 0.0),
        '70r_30s': (0.7, 0.3),
        '50_50': (0.5, 0.5),
        '30r_70s': (0.3, 0.7),
        '100syn': (0.0, 1.0)
    }

    # 1. load all source file lists
    real_io_files = get_image_files(args.real_io_dir)
    real_nio_files = get_image_files(args.real_nio_dir)
    syn_io_files = get_image_files(args.syn_io_dir)
    syn_nio_files = get_image_files(args.syn_nio_dir)

    if not any([real_io_files, real_nio_files, syn_io_files, syn_nio_files]):
        print("Error: No source images found. Please check your input directories.")
        return

    # 2. perform 80/20 train/eval split ONCE for each source folder
    print("Performing 80/20 train-evaluation split on source data...")
    io_real_train, io_real_eval = train_test_split(real_io_files, test_size=0.2, random_state=42)
    nio_real_train, nio_real_eval = train_test_split(real_nio_files, test_size=0.2, random_state=42)
    io_syn_train, io_syn_eval = train_test_split(syn_io_files, test_size=0.2, random_state=42)
    nio_syn_train, nio_syn_eval = train_test_split(syn_nio_files, test_size=0.2, random_state=42)

    # 3. loop through each ratio and create the corresponding dataset folders
    for ratio_str, (real_pct, syn_pct) in ratios.items():
        print(f"\nProcessing ratio: {ratio_str} ({int(real_pct * 100)}% real, {int(syn_pct * 100)}% synthetic)")

        # define directory names
        folder_name_base = f"{ratio_str}_{args.model_id}_{args.product_id}"
        train_dir = os.path.join(args.output_dir, f"train_{folder_name_base}")
        eval_dir = os.path.join(args.output_dir, f"eval_{folder_name_base}")

        train_good_dir = os.path.join(train_dir, "train", "good")
        train_bad_dir = os.path.join(train_dir, "train", "bad")

        # select real and synthetic files for the 'good' (IO) training set
        num_real_io = int(len(io_real_train) * real_pct)
        num_syn_io = int(len(io_syn_train) * syn_pct)
        copy_files(io_real_train[:num_real_io], args.real_io_dir, train_good_dir)
        copy_files(io_syn_train[:num_syn_io], args.syn_io_dir, train_good_dir)

        # select real and synthetic files for the 'bad' (NIO) training set
        num_real_nio = int(len(nio_real_train) * real_pct)
        num_syn_nio = int(len(nio_syn_train) * syn_pct)
        copy_files(nio_real_train[:num_real_nio], args.real_nio_dir, train_bad_dir)
        copy_files(nio_syn_train[:num_syn_nio], args.syn_nio_dir, train_bad_dir)
        print(f"Created training set at: {train_dir}")

        eval_good_dir = os.path.join(eval_dir, "eval", "good")
        eval_bad_dir = os.path.join(eval_dir, "eval", "bad")

        # select real and synthetic files for the 'good' (IO) evaluation set
        num_real_io_eval = int(len(io_real_eval) * real_pct)
        num_syn_io_eval = int(len(io_syn_eval) * syn_pct)
        copy_files(io_real_eval[:num_real_io_eval], args.real_io_dir, eval_good_dir)
        copy_files(io_syn_eval[:num_syn_io_eval], args.syn_io_dir, eval_good_dir)

        # select real and synthetic files for the 'bad' (NIO) evaluation set
        num_real_nio_eval = int(len(nio_real_eval) * real_pct)
        num_syn_nio_eval = int(len(nio_syn_eval) * syn_pct)
        copy_files(nio_real_eval[:num_real_nio_eval], args.real_nio_dir, eval_bad_dir)
        copy_files(nio_syn_eval[:num_syn_nio_eval], args.syn_nio_dir, eval_bad_dir)
        print(f"Created evaluation set at: {eval_dir}")

    print("\nAll datasets created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/eval datasets with specified real-to-synthetic ratios.")

    parser.add_argument("--real-io-dir", type=str, required=True,
                        help="Path to the directory with real 'in order' images.")
    parser.add_argument("--real-nio-dir", type=str, required=True,
                        help="Path to the directory with real 'not in order' images.")
    parser.add_argument("--syn-io-dir", type=str, required=True,
                        help="Path to the directory with synthetic 'in order' images.")
    parser.add_argument("--syn-nio-dir", type=str, required=True,
                        help="Path to the directory with synthetic 'not in order' images.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the generated dataset folders.")
    parser.add_argument("--product-id", type=str, required=True,
                        help="Short identifier for the product (e.g., 'd' for donut).")
    parser.add_argument("--model-id", type=str, required=True, choices=['GAN', 'DM'],
                        help="Identifier for the generative model ('GAN' or 'DM').")  # you can add more models here

    args = parser.parse_args()
    create_datasets(args)
