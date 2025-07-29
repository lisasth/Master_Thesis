"""
This calculates and visualizes a custom cost metric based on the
performance of different trained models. It is designed to evaluate the
economic or operational impact of classification errors (False Positives and
False Negatives) using a predefined cost matrix.

The script performs the following steps:
1.  It iterates through a hardcoded dictionary of model evaluation files,
    each corresponding to a specific real-to-synthetic data ratio.
2.  For each model's evaluation file, it reconstructs the ground truth and
    predicted labels ('good' vs. 'bad') from the per-image results.
3.  It calculates the confusion matrix (TP, FP, FN, TN) for each model's
    performance on the test set.
4.  Using a customizable cost matrix, it computes a total cost score for each
    model, heavily penalizing False Negatives.
5.  Finally, it generates and saves a bar chart that visually compares the
    total cost across all the evaluated models, making it easy to see which
    data composition results in the lowest operational cost.
"""

import os
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix


def compute_cost(tp, fp, fn, tn, cost_matrix):
    """Computes the total cost from a confusion matrix and a cost dictionary."""
    return (
            tp * cost_matrix["TP"] +
            fp * cost_matrix["FP"] +
            fn * cost_matrix["FN"] +
            tn * cost_matrix["TN"]
    )


def main(args):
    """
    Main function to calculate and plot the cost comparison for different models.
    """
    # Define the ratios and display labels
    ratio_map = {
        "100% Synthetic": "100syn",
        "30% Real / 70% Synthetic": "30r_70s",
        "50% Real / 50% Synthetic": "50_50",
        "70% Real / 30% Synthetic": "70r_30s",
        "100% Real": "100real",
    }

    # Construct the cost matrix from arguments
    cost_matrix = {
        "TP": args.cost_tp,
        "FP": args.cost_fp,
        "FN": args.cost_fn,
        "TN": args.cost_tn
    }

    model_costs = {}

    print(f"--- Calculating costs for Product: {args.product}, Model: {args.model_type} ---")

    # Dynamically build paths and process each JSON file
    for display_label, file_label in ratio_map.items():
        filename = f"{file_label}_{args.model_type}.json"
        filepath = os.path.join(args.metrics_root, args.product, filename)

        if not os.path.isfile(filepath):
            print(f"Warning: File not found, skipping: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        y_true, y_pred = [], []
        for image_path, result in data.get("per_image", {}).items():
            true_class = image_path.split("/")[0]
            predicted_class = result.get("prediction", "")
            y_true.append(1 if true_class == "bad" else 0)
            y_pred.append(1 if predicted_class == "bad" else 0)

        if len(set(y_true)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = compute_cost(tp, fp, fn, tn, cost_matrix)
        model_costs[display_label] = total_cost

    if not model_costs:
        print("No data processed. Exiting.")
        return

    # Plotting the results
    plt.figure(figsize=(10, 6))
    labels = list(model_costs.keys())
    values = list(model_costs.values())

    bars = plt.bar(labels, values, color='tomato')
    plt.ylabel("Total Cost")

    # Create a dynamic title
    title = f"Cost Comparison for {args.product.title()} {args.model_type}"
    plt.title(title)
    plt.xticks(rotation=30, ha='right')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Create a dynamic save path
    output_dir = os.path.join(args.plots_root, "costs")
    os.makedirs(output_dir, exist_ok=True)
    save_filename = f"costs_{args.product}_{args.model_type}.png"
    save_path = os.path.join(output_dir, save_filename)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a cost comparison plot from model evaluation results.")

    parser.add_argument("--product", type=str, required=True, choices=['donut', 'croissant'],
                        help="The product to analyze.")
    parser.add_argument("--model-type", type=str, required=True, choices=['GAN', 'DM'],
                        help="The generative model type to analyze.")
    parser.add_argument("--metrics-root", type=str, default="outputs/metrics",
                        help="Root directory of the metrics JSON files.")
    parser.add_argument("--plots-root", type=str, default="outputs/plots",
                        help="Root directory to save the output plots.")
    # Arguments for the cost matrix
    parser.add_argument("--cost-tp", type=int, default=0, help="Cost of a True Positive.")
    parser.add_argument("--cost-fp", type=int, default=1, help="Cost of a False Positive.")
    parser.add_argument("--cost-fn", type=int, default=10, help="Cost of a False Negative.")
    parser.add_argument("--cost-tn", type=int, default=0, help="Cost of a True Negative.")

    args = parser.parse_args()
    main(args)
