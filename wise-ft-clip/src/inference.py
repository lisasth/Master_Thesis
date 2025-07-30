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

# PATH = "/Users/mannich/Documents/Data/Datasets/FnV/FV_PG_Data/val"
PATH = "/Users/mannich/Documents/Data/Datasets/FnV/CCTVMirrorCam"
MODEL_PATH = "/Users/mannich/Documents/Models/FnV/Benjamin/20231221_wise_ft_v2/checkpoint_1.pt"
LABELS_MAPPING_PATH = "/Users/mannich/Documents/Data/Datasets/FnV/FV_PG_Data/val/labels_mapping.csv"
NUM_IMAGES_PER_CLASS = 20


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def wise_ft():

    # Load models
    model = ImageClassifier.load(MODEL_PATH)

    model.eval()

    preprocess = Compose([
        Resize(224),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.5044, 0.4531, 0.4616), (0.1582, 0.1475, 0.1524)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Load labels
    labels = []
    class_ids = []
    with open(LABELS_MAPPING_PATH, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            class_ids.append(int(row[0]))
            labels.append(row[1])

    id_labels_mapping = dict(zip(class_ids, labels))

    images_counter = 0
    top1_counter = 0
    top5_counter = 0

    for subdir, dirs, files in os.walk(PATH):

        article_images_counter = 0
        for file in files:
            if not file.startswith(".") and (file.endswith(".jpeg") or file.endswith(".png")):
                image_path = os.path.join(subdir, file)
                class_id = image_path.split("/")[-2]

                if class_id.isdigit():
                    class_id = int(class_id)
                    image = Image.open(image_path)

                    if not class_id in id_labels_mapping:
                        break

                    image_input = preprocess(image).unsqueeze(0)

                    logits = utils.get_logits(image_input, model)
                    confidences = torch.tensor(logits[-1])
                    probs, indices = torch.topk(confidences, 5)
                    probs = torch.nn.functional.softmax(probs)

                    values = probs.detach().numpy()
                    indices = indices.detach().numpy()

                    print(f"Top 1: GT - {id_labels_mapping[class_id]}, Pred - {labels[indices[0]]}")

                    is_top1 = id_labels_mapping[class_id] == labels[indices[0]]
                    print(f"In Top 1: {is_top1}")

                    is_top5 = id_labels_mapping[class_id] in [labels[idx] for idx in indices]
                    print(f"In Top 5: {is_top5}")

                    print("Top predictions:")
                    for value, index in zip(values, indices):
                        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")

                    images_counter += 1
                    article_images_counter += 1
                    if is_top1:
                        top1_counter += 1

                    if is_top5:
                        top5_counter += 1

                    print(f"Current Top 1 Accuracy: {top1_counter / images_counter}")
                    print(f"Current Top 5 Accuracy: {top5_counter / images_counter}\n")

                    if article_images_counter == NUM_IMAGES_PER_CLASS:
                        article_images_counter = 0
                        break

    print(f"Total Top 1 Accuracy: {top1_counter / images_counter}")
    print(f"Total Top 5 Accuracy: {top5_counter / images_counter}")


if __name__ == '__main__':
    wise_ft()