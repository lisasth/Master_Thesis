import os
import torch
import pickle
from tqdm import tqdm
import math
import numpy as np
import torchvision
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import random
import uuid
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import copy


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class RandAugmentNoColorChange(torchvision.transforms.RandAugment):
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
    ) -> None:
        super().__init__(
            num_ops=num_ops,
            magnitude=magnitude,
            num_magnitude_bins=num_magnitude_bins
        )
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True), # this just adjusts saturation
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            # "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
        }


# def mixup_data(inputs, labels, alpha=1.0):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     print(f"lamda in mixup is {lam}")
#     unique_classes = torch.unique(labels)
#     for unique_class in unique_classes:
#         idx = (labels==unique_class).nonzero()
#         num_sampls = idx.shape[0]
#         if num_sampls > 1:
#             num_samples_to_mix = random.randint(2, num_sampls)
#             index_to_mix = random.sample(idx.squeeze().tolist(), num_samples_to_mix)
#             for j in range(1, len(index_to_mix)):
#                 inputs[index_to_mix[j]] = lam * inputs[index_to_mix[j]] + (1 - lam) * inputs[index_to_mix[j-1]]
#     return inputs
    

# def mixup_data(inputs, labels, alpha=1.0, probability=1.0):
#     if np.random.beta(1.0, 1.0) > probability:
#         return inputs
#     unique_classes = torch.unique(labels)
#     for unique_class in unique_classes:
#         idx = (labels==unique_class).nonzero()
#         num_sampls = idx.shape[0]
#         if num_sampls > 1:
#             lam = np.random.beta(alpha, alpha)
#             # print(f"lamda in mixup is {lam}")
#             index_to_mix = idx.squeeze().tolist()
#             for i in range(0, len(index_to_mix)-1, 2):
#                 idx1, idx2 = index_to_mix[i], index_to_mix[i+1]

#                 inputs[idx1] = lam * inputs[idx1] + (1 - lam) * inputs[idx2]

#                 image_to_save = inputs[idx1].detach().cpu()
#                 mixed_img_save_folder = "mixed_images_medium_aug"
#                 os.makedirs(mixed_img_save_folder, exist_ok=True)
#                 tranformed_img_save_path = os.path.join(mixed_img_save_folder, f"transformed_img_{uuid.uuid1()}.png")
                
#                 save_image(image_to_save, tranformed_img_save_path)
#     return inputs
    

def mixup_data(images, labels, alpha=5.0, probability=0.2):
    if random.uniform(a=0.0, b=1.0) > probability:
        return images
    unique_classes = torch.unique(labels)
    for unique_class in unique_classes:
        idx = (labels==unique_class).nonzero(as_tuple=True)[0]
        if idx.numel()>1:
            perm_idx = idx[torch.randperm(idx.size(0))]
            lam = np.random.beta(alpha, alpha)
            # print(f"lamda in mixup is {lam}")
            images[idx] = lam * images[idx] + (1-lam) * images[perm_idx]

            # image_to_save = images[idx[0]].detach().cpu()
            # mixed_img_save_folder = "mixed_images_medium_aug_4"
            # os.makedirs(mixed_img_save_folder, exist_ok=True)
            # tranformed_img_save_path = os.path.join(mixed_img_save_folder, f"transformed_img_{uuid.uuid1()}.png")
            
            # save_image(image_to_save, tranformed_img_save_path)

    return images


def mixup_data_different_classes(images, labels, alpha=5.0, probability=0.2):
    batch_size = images.size(0)
    mixed_images = images.clone()
    mixed_binary_labels = torch.zeros(batch_size, dtype=torch.long)
    
    num_classes = torch.max(labels) + 1
    labels_one_hot = F.one_hot(labels, num_classes).float()
    mixed_labels = labels_one_hot.clone()
    
    for i in range(batch_size):
        if random.uniform(0.0, 1.0) <= probability:
            # Find indices of images from different classes
            different_class_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
            
            if different_class_indices.numel() > 0:
                # Choose a random image from a different class
                j = different_class_indices[torch.randint(0, different_class_indices.size(0), (1,)).item()]
                
                # Apply mixup
                lam = np.random.beta(alpha, alpha)
                mixed_images[i] = lam * images[i] + (1 - lam) * images[j]
                mixed_labels[i] = lam * labels_one_hot[i] + (1 - lam) * labels_one_hot[j]
                mixed_binary_labels[i] = 1
                
                # Save mixed images
                image_to_save = mixed_images[i].detach().cpu()
                mixed_img_save_folder = "mixed_images_differ_classes"
                os.makedirs(mixed_img_save_folder, exist_ok=True)
                tranformed_img_save_path = os.path.join(mixed_img_save_folder, f"transformed_img_{uuid.uuid1()}.png")
                
                save_image(image_to_save, tranformed_img_save_path)
    
    return mixed_images, mixed_labels, mixed_binary_labels


def get_last_linear_layer_info(model):
    """
    Finds and returns the last fully connected (linear) layer of the given model along with its attribute path.
    """
    # Check for common attribute names
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return 'fc', model.fc
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            last_layer = list(model.classifier.children())[-1]
            
            if isinstance(last_layer, nn.Linear):
                return ('classifier', -1), last_layer  # Sequential index
            
        elif isinstance(model.classifier, nn.Linear):
            return 'classifier', model.classifier
    elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        return 'head', model.head

    # Fallback: Iterate through all modules
    for name, module in reversed(model._modules.items()):
        if isinstance(module, nn.Linear):
            return name, module
    
    return None, None


def replace_last_linear_layer(model, old_last_layer_info, new_last_layer):
    """
    Replaces the last linear layer of the model with a new_last_layer.
    
    """
    # Update the model with the new layer
    if isinstance(old_last_layer_info, tuple):  # Sequential case, like ('fc', -1)
        getattr(model, old_last_layer_info[0])[old_last_layer_info[1]] = new_last_layer
    else:
        setattr(model, old_last_layer_info, new_last_layer)
    
    return model


def get_entropy_loss(args):
    if args.entropy_loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif args.entropy_loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        raise Exception("Unsupported entropy loss, please choose between 'CrossEntropyLoss' and 'BCEWithLogitsLoss'") 

def compute_entropy_loss(loss_func_entropy, logits, labels, num_classes, device):
    if isinstance(loss_func_entropy, nn.CrossEntropyLoss):
        loss_entropy = loss_func_entropy(logits, labels)
    elif isinstance(loss_func_entropy, nn.BCEWithLogitsLoss):
        labels = get_one_hot_labels(copy.deepcopy(labels), num_classes, device)
        loss_entropy = loss_func_entropy(logits, labels)
    return loss_entropy


def get_one_hot_labels(labels, num_classes, device):
    labels_one_hot = torch.zeros(labels.size(0), num_classes).to(device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot


def two_head_forward(model, batch, loss_func_entropy, loss_func_contra, num_classes, device):
    inputs = batch["images"]
    labels = batch['labels'].to(device)
    images = torch.cat([inputs[0], inputs[1]], dim=0).to(device)
    batch_size = labels.shape[0]
    pred_supcon, logits = model(images)
    f1, f2 = torch.split(pred_supcon, [batch_size, batch_size], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss_contra = loss_func_contra(features, labels)
    labels = torch.cat((labels, labels))
    loss_entropy = compute_entropy_loss(loss_func_entropy, logits, labels, num_classes, device)
    return loss_contra, loss_entropy, logits, labels


def one_head_forward(model, batch, loss_func_entropy, num_classes, device):
    inputs = batch["images"].to(device)
    logits = model(inputs)
    labels = batch['labels'].to(device)
    loss_entropy = compute_entropy_loss(loss_func_entropy, logits, labels, num_classes, device)
    return loss_entropy, logits, labels

def get_backbone_and_embedding_dim(model_type, model_path):
    if model_type == "clip":
        model = torch_load(model_path)
        backbone = model.image_encoder
        embedding_dim = backbone.model.ln_final.normalized_shape[0]
    elif model_type == "efficientnet":
        model = torch_load(model_path)
        backbone = nn.Sequential(model.features, model.avgpool)
        embedding_dim = (list(model.features.children())[-1]).out_channels
    else:
        raise ValueError("Unsupported model type, please choose in ['clip', 'efficientnet']")
    
    return backbone, embedding_dim

def attach_head_to_model(model_type, class_head, model):
    if model_type == "clip":
        model.classification_head = class_head
    elif model_type == "efficientnet":
        model.classifier[-1] = class_head
    else:
        raise ValueError("Unsupported model type, please choose in ['clip', 'efficientnet']")
    
    return model

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x