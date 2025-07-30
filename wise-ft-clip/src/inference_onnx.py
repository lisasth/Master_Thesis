import csv
import os

from PIL import Image

import clip
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from scipy.special import softmax

from src.models import utils
from src.models.modeling import ImageClassifier
import onnxruntime
from sklearn.metrics import accuracy_score
from tqdm import tqdm

DATA_PATH = "/home/jingjie/wise-ft-clip/data/Eval_UK_cleaned"
MODEL_PATH = "/home/jingjie/atalos/tmp/Models/product_class_pos/EVA02-CLIP-B-16_fnv_head_cleaned_k_data_0506"
LABELS_FILE_PATH = "/home/jingjie/wise-ft-clip/class_names_list/classes_2024_03_26_uk.txt"


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def top_k_acc(val_gts, preds_logits):
    top_4_predictions = np.argsort(preds_logits, axis=1)[:, -4:]
    num_samples = len(top_4_predictions)
    num_correct = 0
    for i in range(num_samples):
        if val_gts[i] in top_4_predictions[i]:
            num_correct += 1
    top_4_acc = num_correct / num_samples
    return top_4_acc

def inference_onnx():

    # Load models
    models = sorted(entry.name for entry in os.scandir(MODEL_PATH) if entry.is_file())
    for model in models:
        ort_session = onnxruntime.InferenceSession(os.path.join(MODEL_PATH, model), providers=['CPUExecutionProvider'])

        preprocess = Compose([
            Resize([224,224]),
            # CenterCrop(224),
            # _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        # Load labels
        class_name_to_idx_mapping = {}
        with open(LABELS_FILE_PATH, "r") as f:
            for index, line in enumerate(f):
                class_name_to_idx_mapping[line.rstrip()] = index

        val_gts_onnx_all = []
        val_preds_onnx_all = []
        val_scores_onnx_all = []
        for entry in os.scandir(DATA_PATH):
            if entry.is_dir():
                single_dataset_path = os.path.join(DATA_PATH, entry)

                val_gts_onnx = []
                val_preds_onnx = []
                val_scores_onnx = []

                for subdir, _, files in os.walk(single_dataset_path):
                    for file in files:
                        if not file.startswith(".") and (file.endswith(".jpeg") or file.endswith(".png")):
                            image_path = os.path.join(subdir, file)
                            gt_class_name = image_path.split("/")[-2]

                            image = Image.open(image_path)

                            if not gt_class_name in class_name_to_idx_mapping.keys():
                                break

                            image_input = preprocess(image).unsqueeze(0)
                            ort_inputs = {ort_session.get_inputs()[0].name: image_input.detach().cpu().numpy()}
                            a = ort_session.run(None, ort_inputs)
                            logits_onnx = ort_session.run(None, ort_inputs)[1]
                            pred_onnx = np.argmax(logits_onnx, axis=1)
                            val_gts_onnx.append(class_name_to_idx_mapping[gt_class_name])
                            val_preds_onnx.extend(pred_onnx)
                            val_scores_onnx.append(logits_onnx)

                preds_logits = np.concatenate(val_scores_onnx, axis=0)
                top_4_acc_onnx = top_k_acc(val_gts_onnx, preds_logits)
                val_acc_onnx = accuracy_score(val_gts_onnx, val_preds_onnx)
                print(f"model:", model)
                print(f"val_acc_onnx in {single_dataset_path}: {val_acc_onnx:.4f}")
                print(f"val_top_4_acc_onnx in {single_dataset_path}: {top_4_acc_onnx:.4f}")
                val_gts_onnx_all.extend(val_gts_onnx)
                val_preds_onnx_all.extend(val_preds_onnx)
                val_scores_onnx_all.extend(val_scores_onnx)
        
        preds_logits = np.concatenate(val_scores_onnx_all, axis=0)
        top_4_acc_onnx = top_k_acc(val_gts_onnx_all, preds_logits)
        val_acc_onnx = accuracy_score(val_gts_onnx_all, val_preds_onnx_all)
        print(f"model:", model)
        print(f"val_acc_onnx in {DATA_PATH}: {val_acc_onnx:.4f}")
        print(f"val_top_4_acc_onnx in {DATA_PATH}: {top_4_acc_onnx:.4f}")
        print("------------------------------------------------------------------")



if __name__ == '__main__':
    inference_onnx()