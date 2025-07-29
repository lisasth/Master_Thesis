"""
This provides an automated pipeline for evaluating a series of pre-trained
image classifiers. It systematically loads each model, runs it on a standardized
test dataset, and computes a comprehensive set of performance metrics.

The script is designed to be run from the command line, allowing for flexible
specification of the product, model directory, and test data directory.
e.g. python evaluation/utility/utility_assessment.py --test-dir data/test/croissant --product croissant

The main workflow is as follows:
1.  Iterate Through Experiments: The script loops through all combinations of
    predefined data ratios (e.g., '100real', '70r_30s') and model types
    (e.g., 'GAN', 'DM').
2.  Load Model: For each combination, it constructs the path to the corresponding
    trained model checkpoint (`.pt` file) and loads the model along with its
    specific preprocessing pipeline.
3.  Perform Inference: It processes every image in the provided test directory,
    which is expected to contain 'good' and 'bad' subfolders representing the
    two classes. It stores the true labels and the model's predicted probabilities.
4.  Calculate Metrics: After evaluating all test images, it uses scikit-learn
    to compute a full suite of classification metrics, including accuracy,
    precision, recall, F1-score, ROC-AUC, PR-AUC, and a confusion matrix.
5.  Save Results: The script saves the computed metrics and detailed per-image
    results into a structured JSON file, creating one output file for each
    model evaluated. These JSON files serve as the input for subsequent analysis
    and visualization scripts.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from evaluation.models.modeling import ImageClassifier


def evaluate_model(model_path, test_dir, output_json_path):
    """
    Loads a trained model, evaluates it on a test set, and saves the metrics.
    """
    print(f"\nEvaluating Model: {os.path.basename(model_path)}")
    print(f"Test data: {test_dir}")

    # 1. load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For Apple Silicon, uncomment the line below
    # device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        model = ImageClassifier.load(model_path).to(device)
        model.eval()
        preprocess = model.val_preprocess
    except Exception as e:
        print(f"Error loading model {model_path}: {e}. Skipping.")
        return

    # 2. prepare for evaluation
    class_map = {"good": 0, "bad": 1}
    inverse_map = {v: k for k, v in class_map.items()}
    y_true, y_pred, y_scores, results, wrong_images = [], [], [], {}, []

    # 3. iterate through test data and make predictions
    for true_class in class_map:
        folder = os.path.join(test_dir, true_class)
        if not os.path.isdir(folder):
            print(f"Warning: Test subfolder not found: {folder}")
            continue

        for img_name in os.listdir(folder):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(folder, img_name)
            image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = logits.softmax(dim=-1).cpu().numpy().flatten()

            pred_class_id = int(np.argmax(probs))
            true_class_id = class_map[true_class]

            y_true.append(true_class_id)
            y_pred.append(pred_class_id)
            y_scores.append(probs)

            if pred_class_id != true_class_id:
                wrong_images.append(f"{true_class}/{img_name}")

            results[f"{true_class}/{img_name}"] = {
                "prediction": inverse_map[pred_class_id],
                "probabilities": {
                    inverse_map[0]: float(probs[0]),
                    inverse_map[1]: float(probs[1])
                }
            }

    if not y_true:
        print("Error: No test images were processed. Skipping metric calculation.")
        return

    # 4. compute metrics
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_scores[:, 1])
        pr_auc = average_precision_score(y_true, y_scores[:, 1])
        conf_matrix = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Could not compute all metrics: {e}. Saving partial results.")
        summary = {"error": str(e)}
    else:
        summary = {
            "accuracy": accuracy,
            "precision_per_class": {inverse_map.get(i, f'class_{i}'): float(p) for i, p in enumerate(precision)},
            "recall_per_class": {inverse_map.get(i, f'class_{i}'): float(r) for i, r in enumerate(recall)},
            "f1_score_per_class": {inverse_map.get(i, f'class_{i}'): float(f) for i, f in enumerate(f1)},
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": conf_matrix.tolist(),
            "wrong_images": wrong_images
        }

    # 5. save results
    output = {"summary": summary, "per_image": results}
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_json_path}")
    print(f"Accuracy: {summary.get('accuracy', 'N/A'):.3f}, ROC-AUC: {summary.get('roc_auc', 'N/A'):.3f}")

    return summary


def main(args):
    ratios = ['100real', '70r_30s', '50_50', '30r_70s', '100syn']
    model_types = ['GAN', 'DM']

    product_char_map = {
        'donut': 'd',
        'croissant': 'c',
        'cashew': 'cashew'
    }
    product_char = product_char_map.get(args.product)
    if not product_char:
        raise ValueError(f"Product '{args.product}' not recognized. Add it to product_char_map.")

    all_results = []

    for ratio in ratios:
        for model_type in model_types:
            model_filename = f"{ratio}_{model_type}_{product_char}.pt"
            model_path = os.path.join(args.models_dir, model_filename)

            output_filename = f"{ratio}_{model_type}.json"
            output_json_path = os.path.join(args.metrics_dir, args.product, output_filename)
            summary = evaluate_model(model_path, args.test_dir, output_json_path)

            if summary and "error" not in summary:
                row_data = {
                    "product": args.product,
                    "ratio": ratio,
                    "model_type": model_type,
                    "accuracy": summary.get("accuracy"),
                    "roc_auc": summary.get("roc_auc"),
                    "pr_auc": summary.get("pr_auc"),
                }
                # Unpack per-class metrics into their own columns for a cleaner CSV
                for metric in ['precision', 'recall', 'f1_score']:
                    for class_name, value in summary.get(f'{metric}_per_class', {}).items():
                        row_data[f'{metric}_{class_name}'] = value

                all_results.append(row_data)

    if not all_results:
        print("\nNo models were successfully evaluated. Skipping CSV creation.")
        return

    print("\n" + "=" * 50)
    print("Aggregating results into a single CSV file...")

    results_df = pd.DataFrame(all_results)

    # Ensure a consistent column order
    column_order = [
        "product", "ratio", "model_type", "accuracy", "roc_auc", "pr_auc",
        "precision_good", "precision_bad", "recall_good", "recall_bad",
        "f1_score_good", "f1_score_bad"
    ]
    # Filter for columns that actually exist in the DataFrame
    final_columns = [col for col in column_order if col in results_df.columns]
    results_df = results_df[final_columns]

    # Define and save the CSV file
    output_csv_path = os.path.join(args.metrics_dir, args.product, f"{args.product}_summary_all_models.csv")
    results_df.to_csv(output_csv_path, index=False, float_format='%.4f')

    print(f"✅ Successfully created aggregated summary: {output_csv_path}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated evaluation of trained image classifiers.")

    parser.add_argument("--models-dir", type=str, default="models",
                        help="Base directory where trained model folders are saved.")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to the test dataset directory (containing 'good' and 'bad' subfolders).")
    parser.add_argument("--metrics-dir", type=str, default="./outputs/metrics",
                        help="Base directory to save the output JSON metrics files.")
    parser.add_argument("--product", type=str, required=True, choices=['donut', 'croissant', 'cashew'],
                        help="The product being evaluated.")

    args = parser.parse_args()
    main(args)
