import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.args import parse_arguments
from torchvision.transforms import transforms
import src.datasets as datasets
from src.datasets.common import maybe_dictionarize, get_dataset, TwoHead
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io
from src.models import utils
from PIL import Image
from src.models.modeling import TwoViewTransform
from src.models.contrastive_learn import SupConLoss
import copy
from src.models.early_stopping import EarlyStopping




def _convert_to_rgb(image):
    return image.convert('RGB')

def build_transform(n_px: int, is_train: bool):

    custom_transforms = [ # saturation0.6 aug
        # transforms.ElasticTransform(),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.25),
        transforms.RandomAutocontrast(p=0.25),
        # transforms.RandomEqualize(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=[0.6,1.1]),
    ]
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return transforms.Compose([
            *custom_transforms,
            # RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
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
        one_hot_labels = get_one_hot_labels(copy.deepcopy(labels), num_classes, device)
        loss_entropy = loss_func_entropy(logits, one_hot_labels)
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

def main(args):
    class_in_all_product_to_idx = {}
    with open(args.classes_list_path, "r") as f:
        for index, line in enumerate(f):
            class_in_all_product_to_idx[line.rstrip()] = index
    all_classes_num = len(class_in_all_product_to_idx)

    # Load the pretrained Pytorch model
    if args.model_type == 'efficientnet':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    elif args.model_type == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
    num_classes = all_classes_num
    old_last_layer_info, last_linear_layer = utils.get_last_linear_layer_info(model)
    new_linear_layer = nn.Linear(last_linear_layer.in_features, num_classes)
    model = utils.replace_last_linear_layer(model, old_last_layer_info, new_linear_layer)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    if args.contrastive:
        old_last_layer_info, last_linear_layer = utils.get_last_linear_layer_info(model)
        new_last_layer = TwoHead(last_linear_layer)
        model = utils.replace_last_linear_layer(model, old_last_layer_info, new_last_layer)
        loss_func_contra = SupConLoss(temperature=args.temperature)
        print("Train with contrastive learning")


    if args.train_dataset == "SampleFileLoader":
        args.train_sample_file_path = os.path.join(args.train_sample_files_folder_path, "overall_balanced.json")
    train_dataset = get_dataset(args=args, is_train=True)
    train_loader = train_dataset.data_loader

    if args.eval_datasets == "SampleFileLoader":
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, "overall_balanced.json")
    val_dataset = get_dataset(args, is_train=False)
    val_loader = val_dataset.data_loader

    loss_func_entropy = get_entropy_loss(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    writer = SummaryWriter(log_dir=args.logdir)
    iteration = 0

    #Make sure one model is always saved
    if args.early_stop_min_iterations > len(train_loader)*args.epochs:
        early_stop_min_iterations = len(train_loader)-1
        print(f"The parameter early_stop_min_iterations was set too high to save any models. Setting it to {early_stop_min_iterations}")
    else:
        early_stop_min_iterations = args.early_stop_min_iterations 
    
    early_stop = EarlyStopping(save_path=args.save, patience=args.early_stop_patience, verbose=True,
                               min_iteration=early_stop_min_iterations)

    num_samples_iterations = 0.
    correct_samples_iterations = 0.
    train_loss_iterations = 0.
    print_every = 1


    for epoch in range(args.epochs):
        if early_stop.early_stop_flag:
            break
        for i, batch in enumerate(train_loader):
            model = model.to(device)
            model.train()
            batch = maybe_dictionarize(batch)

            if args.contrastive:
                loss_contra, loss_entropy, logits, labels = two_head_forward(model, batch, loss_func_entropy, loss_func_contra, num_classes, device)
                loss = args.contra_loss_weight * loss_contra + (1 - args.contra_loss_weight) * loss_entropy
            else:
                loss_entropy, logits, labels = one_head_forward(model, batch, loss_func_entropy, num_classes, device)
                loss = loss_entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct_num_batch = (preds == labels).float().sum()
            all_num_batch = len(preds)
            num_samples_iterations += all_num_batch
            correct_samples_iterations += correct_num_batch
            train_loss_iterations += loss

            if i % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.4f}\t Acc: {correct_num_batch/all_num_batch:.4f}\t", flush=True
                )
            

            
            if iteration % args.log_frequency == 0:
                train_acc = correct_samples_iterations/num_samples_iterations
                train_loss = train_loss_iterations / num_samples_iterations
                writer.add_scalar('Acc_i/train', train_acc, iteration)
                writer.add_scalar('Loss_i/train', train_loss, iteration)
                print(
                    f"Iteration: {iteration}\t"
                    f"Train loss: {train_loss:.4f}\t Train acc: {train_acc:.4f}\t", flush=True
                )
                num_samples_iterations = 0.
                correct_samples_iterations = 0.
                train_loss_iterations = 0.

                # Validation Phase
                print("Starting evaluation")
                model.eval()
                val_loss = 0.0
                num_samples_val = 0.
                correct_samples_val = 0.

                with torch.no_grad():
                    for batch in val_loader:
                        batch = maybe_dictionarize(batch)
                        inputs = batch["images"].to(device)
                        if args.contrastive:
                            _, logits = model(inputs)
                        else:
                            logits = model(inputs)
                        labels = batch['labels'].to(device)
                        loss = compute_entropy_loss(loss_func_entropy, logits, labels, num_classes, device)

                        preds = torch.argmax(logits, dim=1)
                        correct_num_batch = (preds == labels).float().sum()
                        all_num_batch = len(preds)
                        num_samples_val += all_num_batch
                        correct_samples_val += correct_num_batch
                        val_loss += loss
                val_acc = correct_samples_val/num_samples_val
                val_loss = val_loss / num_samples_val
                writer.add_scalar('Acc_i/val', val_acc, iteration)
                writer.add_scalar('Loss_i/val', val_loss, iteration)
                print(
                    f"Iteration: {iteration}\t"
                    f"Eval loss: {val_loss:.4f}\t Eval acc: {val_acc:.4f}\t", flush=True
                )

                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    model_to_save = copy.deepcopy(model)
                    if args.contrastive:
                        model_to_save.classifier[-1] = model_to_save.classifier[-1].linear_layer
                    early_stop(val_acc, model_to_save, iteration)
                    if early_stop.early_stop_flag:
                        print("Early stopping")
                        break

            iteration += 1


if __name__ == '__main__':
    args = parse_arguments()
    main(args)