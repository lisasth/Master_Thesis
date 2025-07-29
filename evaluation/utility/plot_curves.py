"""
This generates ROC (Receiver Operating Characteristic) and Precision-Recall (PR) curves
to visually compare the performance of different classifier models.

It operates by performing the following steps:
1.  It loads a predefined set of JSON files, where each file contains the
    detailed per-image prediction results (true class vs. predicted probabilities)
    for a specific model.
2.  For each model, it extracts the ground truth labels and the corresponding
    prediction scores for the positive class ('bad').
3.  It uses these labels and scores to calculate the points required to plot
    the ROC curve (False Positive Rate vs. True Positive Rate) and the
    Precision-Recall curve.
4.  Finally, it generates two separate plots: one overlaying the ROC curves for
    all models and another overlaying the PR curves. These plots are then
    saved as PNG files to a specified output directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve

def main(args):
    """
    Main function to load data, calculate ROC/PR curves, and generate plots.
    """
    # Map display labels to the filenames for different ratios
    ratio_map = {
        "100% Synthetic": "100syn",
        "30% Real / 70% Synthetic": "30r_70s",
        "50% Real / 50% Synthetic": "50_50",
        "70% Real / 30% Synthetic": "70r_30s",
        "100% Real": "100real",
    }

    # Dynamically build the list of files to process
    model_files = {}
    for display_label, file_label in ratio_map.items():
        filename = f"{file_label}_{args.model_type}.json"
        path = os.path.join(args.metrics_root, args.product, filename)
        model_files[display_label] = path

    roc_curves = {}
    pr_curves = {}

    print(f"--- Processing curves for Product: {args.product}, Model: {args.model_type} ---")
    for model_name, filepath in model_files.items():
        if not os.path.isfile(filepath):
            print(f"Warning: File not found, skipping: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        per_image = data.get("per_image", {})
        y_true, y_scores = [], []

        # This logic for extracting labels is a bit complex; simplifying it.
        for image_path, result in per_image.items():
            true_class = image_path.split("/")[0]
            score = result.get("probabilities", {}).get("bad", 0.0)

            y_true.append(1 if true_class == "bad" else 0)
            y_scores.append(score)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            roc_curves[model_name] = (fpr, tpr)
            pr_curves[model_name] = (recall, precision)

    if not roc_curves:
        print("No data was processed to generate curves. Exiting.")
        return

    # A map for creating concise filenames
    product_char_map = {'donut': 'd', 'croissant': 'c'}
    product_char = product_char_map.get(args.product, 'x')

    # Create dynamic save paths
    output_dir = os.path.join(args.plots_root, args.product, args.model_type)
    os.makedirs(output_dir, exist_ok=True)

    roc_save_path = os.path.join(output_dir, f"roc_curves_{product_char}_{args.model_type}.png")
    pr_save_path = os.path.join(output_dir, f"pr_curves_{product_char}_{args.model_type}.png")

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves {args.product.title()} {args.model_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_save_path, dpi=300)
    plt.close()
    print(f"\nROC curve plot saved to: {roc_save_path}")

    # Plot PR curves
    plt.figure(figsize=(8, 6))
    for model_name, (recall, precision) in pr_curves.items():
        plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves {args.product.title()} {args.model_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pr_save_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curve plot saved to: {pr_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ROC and Precision-Recall curves from model evaluation files.")

    parser.add_argument("--product", type=str, required=True, choices=['donut', 'croissant'],
                        help="The product to analyze.")
    parser.add_argument("--model-type", type=str, required=True, choices=['GAN', 'DM'],
                        help="The generative model type to analyze.")
    parser.add_argument("--metrics-root", type=str, default="outputs/metrics",
                        help="Root directory of the metrics JSON files.")
    parser.add_argument("--plots-root", type=str, default="outputs/plots",
                        help="Root directory to save the output plots.")

    args = parser.parse_args()
    main(args)