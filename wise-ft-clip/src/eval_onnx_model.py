import os
import numpy as np
import torch 
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize, get_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import onnxruntime


def calculate_top_4_acc(val_gts, preds_logits):
    top_4_predictions = np.argsort(preds_logits, axis=1)[:, -4:]
    num_samples = len(top_4_predictions)
    num_correct = 0
    for i in range(num_samples):
        if val_gts[i] in top_4_predictions[i]:
            num_correct += 1
    top_4_acc = num_correct / num_samples
    return top_4_acc


def _convert_to_rgb(image):
    return image.convert('RGB')

def build_transform(n_px: int, is_train: bool):

    custom_transforms = [
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.25),
        transforms.RandomAutocontrast(p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=[0.6,1.1]),
    ]
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return transforms.Compose([
            *custom_transforms,
            transforms.RandomResizedCrop(n_px, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])

class ExtendedLinearClassifier(nn.Linear):
    def __init__(self, old_classifier, new_classes=1):
        self.old_weights = old_classifier.weight.detach().clone()
        self.old_bias = old_classifier.bias.detach().clone()
        super().__init__(self.old_weights.size(1), self.old_weights.size(0) + new_classes)
        with torch.no_grad():
            self.weight[:self.old_weights.size(0), :] = self.old_weights
            self.bias[:self.old_bias.size(0)] = self.old_bias

    # Freeze gradients for old classes
    def freeze_old_classes_grad(self):
        if self.weight.grad is not None:
            self.weight.grad[:self.old_weights.size(0), :] = 0
        if self.bias.grad is not None:
            self.bias.grad[:self.old_bias.size(0)] = 0



def eval_add_class_into_model(args):

    model_to_eval = args.load

    all_preds = []
    all_targets = []
    val_scores = []
    num_samples_val = 0.
    correct_samples_val = 0.

    ort_session = onnxruntime.InferenceSession(model_to_eval, providers=['CUDAExecutionProvider'])


    if args.eval_datasets == "SampleFileLoader":
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, "overall_balanced.json")
    val_dataset = get_dataset(args, is_train=False)
    val_loader = val_dataset.data_loader

    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].detach().cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: inputs}
            outputs = ort_session.run(None, ort_inputs)[0]
            labels = batch['labels'].detach().cpu().numpy()
            preds = np.argmax(outputs, axis=1)

            correct_num_batch = np.sum(preds == labels)
            all_num_batch = len(preds)
            num_samples_val += all_num_batch
            correct_samples_val += correct_num_batch

            all_preds.extend(preds)
            all_targets.extend(labels)
            val_scores.append(outputs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.concatenate(val_scores, axis=0)

    # Calculate accuracy, F1 score, and confusion matrix
    accuracy_manual = correct_samples_val/num_samples_val
    accuracy = accuracy_score(all_targets, all_preds)
    top_4_acc = calculate_top_4_acc(all_targets, all_logits)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    idx_to_class_name = {}
    with open(args.classes_list_path, "r") as f:
        for index, line in enumerate(f):
            idx_to_class_name[index] = line.rstrip()
    labels = [i for i in range(len(idx_to_class_name))]
    target_names = [idx_to_class_name[idx] for idx in labels]
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot()
    fig, ax = plt.subplots(figsize=(25, 25))
    cm_display.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.tight_layout()
    # save_cm_path = os.path.join(args.save, "confusion_matrix" + ".png")
    # plt.savefig(save_cm_path, dpi=300)
    print(classification_report(all_targets, all_preds, target_names=target_names, labels=labels))

    # Print results
    print(f'Accuracy: {accuracy_manual:.4f}')
    print(f'Accuracy_manual_cal: {accuracy:.4f}')
    print(f'Top4 Accuracy: {top_4_acc:.4f}')
    print(f'F1 Score (weighted): {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)


if __name__ == '__main__':
    args = parse_arguments()
    eval_add_class_into_model(args)

