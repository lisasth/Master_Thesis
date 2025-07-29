"""
This serves as a visualization tool for plotting various model performance metrics
that have been previously computed and stored in JSON files.

The script is designed to be run from the command line, allowing the user to
specify which product (e.g., 'donut', 'croissant') and which model type
(e.g., 'DM', 'GAN') to generate plots for.

It performs the following steps:
1.  It dynamically constructs the paths to the relevant JSON summary files based
    on the provided command-line arguments.
2.  It loads the data for a predefined set of metrics (accuracy, ROC-AUC, F1-score,
    precision, and recall) from the JSON files.
3.  For each metric, it generates a separate bar chart that compares the performance
    across different real-to-synthetic data ratios.
4.  Each plot is given a dynamic title and saved to a structured output directory,
    organized by product and model type, for easy access and comparison.
"""

import matplotlib.pyplot as plt
import json
import os
import argparse


def plot_metric(metric_dict, title, ylabel, save_path):
    """
    Generates and saves a bar chart for a given metric.
    """
    labels = list(metric_dict.keys())
    values = list(metric_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0.1, 1.05)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{value:.3f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")


def main(args):
    ratio_map = {
        "100% Synthetic": "100syn",
        "30% Real / 70% Synthetic": "30r_70s",
        "50% Real / 50% Synthetic": "50_50",
        "70% Real / 30% Synthetic": "70r_30s",
        "100% Real": "100real",
    }

    # dictionary to hold all metric data
    metrics = {
        "accuracy": {}, "roc_auc": {}, "pr_auc": {},
        "f1_score_good": {}, "f1_score_bad": {},
        "precision_good": {}, "precision_bad": {},
        "recall_good": {}, "recall_bad": {}
    }

    print(f"Loading data for Product: {args.product.title()}, Model: {args.model_type}")
    for display_label, file_label in ratio_map.items():
        filename = f"{file_label}_{args.model_type}.json"
        path = os.path.join(args.metrics_root, args.product, filename)

        if not os.path.isfile(path):
            print(f"Warning: File not found, skipping: {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)
            summary = data.get("summary", {})
            # metrics dict
            metrics["accuracy"][display_label] = summary.get("accuracy", 0)
            metrics["roc_auc"][display_label] = summary.get("roc_auc", 0)
            metrics["pr_auc"][display_label] = summary.get("pr_auc", 0)
            metrics["f1_score_good"][display_label] = summary.get("f1_score_per_class", {}).get("good", 0)
            metrics["f1_score_bad"][display_label] = summary.get("f1_score_per_class", {}).get("bad", 0)
            metrics["precision_good"][display_label] = summary.get("precision_per_class", {}).get("good", 0)
            metrics["precision_bad"][display_label] = summary.get("precision_per_class", {}).get("bad", 0)
            metrics["recall_good"][display_label] = summary.get("recall_per_class", {}).get("good", 0)
            metrics["recall_bad"][display_label] = summary.get("recall_per_class", {}).get("bad", 0)

    # map for creating filenames
    product_char_map = {'donut': 'd', 'croissant': 'c', 'cashew': 'cashew'}
    product_char = product_char_map.get(args.product, 'x')

    # generate plot for each metric
    for metric_name, metric_data in metrics.items():
        if not metric_data:
            continue

        title = f"{metric_name.replace('_', ' ').title()} Comparison {args.product.title()} {args.model_type}"
        ylabel = metric_name.replace('_', ' ').title()
        save_dir = os.path.join(args.plots_root, args.product, args.model_type)
        filename = f"{metric_name}_{args.model_type}_{product_char}.png"
        save_path = os.path.join(save_dir, filename)
        plot_metric(metric_data, title, ylabel, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for model evaluation metrics from JSON files.")
    parser.add_argument("--product", type=str, required=True, choices=['donut', 'croissant', 'cashew'],
                        help="Product to analyze (e.g., 'donut').")
    parser.add_argument("--model-type", type=str, required=True, choices=['DM', 'GAN'],
                        help="Model type to analyze ('DM' or 'GAN').")
    parser.add_argument("--metrics-root", type=str, default="./outputs/metrics",
                        help="Root directory of the metrics JSON files.")
    parser.add_argument("--plots-root", type=str, default="./outputs/plots",
                        help="Root directory to save the output plots.")

    args = parser.parse_args()
    main(args)
