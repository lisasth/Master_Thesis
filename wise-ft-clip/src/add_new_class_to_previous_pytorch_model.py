import os
import numpy as np
import torch
import torch 
import torch.nn as nn
import torch.optim as optim

from src.models.eval import evaluate
from src.models.camptum_visual import camptum_visual
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load, torch_load
from src.models import utils
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import src.datasets as datasets
from torchvision.transforms import transforms
from PIL import Image
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io
from tqdm import tqdm


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
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=[0.6, 1.1]),
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

class ExtendedLinearClassifier(nn.Linear): # Shoulbe be changed to be subclass of torch.nn.Linear
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


def add_class_into_model(args):
    

    # Load models
    model = torch_load(args.load)
    model.eval()

    if args.add_new_class:
        model.fc = ExtendedLinearClassifier(old_classifier=model.fc)

    # Freeze backbone
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    train_preprocess_fn = build_transform(n_px=224, is_train=True)
    dataset_class = getattr(datasets, args.dataset)
    class_scanned_percentage_path = None
    if args.class_scanned_percentage_path is not None:
        class_scanned_percentage_path = args.class_scanned_percentage_path
    dataset = dataset_class(
        train_set=args.train_set,
        eval_set=args.eval_set,
        preprocess=train_preprocess_fn,
        classes_list_path=args.classes_list_path,
        location=args.data_location,
        batch_size=args.batch_size,
        weighted_sampler=args.weighted_sampler,
        class_scanned_percentage_path=class_scanned_percentage_path,
    )
    # Modify the label to be 0 if not the new class, to be 1 if it's the new class
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.wd) # Optimize only the new classifier's weights and biases
    writer = SummaryWriter(log_dir=args.logdir)
    iteration = 0

    # Determine the label for the new class, new class must be given
    single_trained_class_label = args.single_trained_class_index - 1


    num_samples_iterations = 0.
    correct_samples_iterations = 0.
    train_loss_iterations = 0.
    print_every = 1
    for epoch in range(args.epochs):
        data_loader = dataset.train_loader
        for i, batch in enumerate(data_loader):
            model = model.cuda()
            model.fc.train()  # Set only model.fc to train to make sure the BN will not change
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            outputs = model(inputs)
            labels = batch['labels'].cuda()
            new_class_targets = torch.where(labels == single_trained_class_label, 1.0, 0.0).float()
            loss = criterion(outputs[:, single_trained_class_label], new_class_targets)

            optimizer.zero_grad()
            loss.backward()
            # Freeze the old class weights
            # model.fc.freeze_old_classes_grad()
            optimizer.step()

            preds = (torch.sigmoid(outputs[:, single_trained_class_label]) > 0.5).float()
            correct_num_batch = (preds == new_class_targets).float().sum()
            all_num_batch = len(preds)
            num_samples_iterations += all_num_batch
            correct_samples_iterations += correct_num_batch
            train_loss_iterations += loss

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.4f}\t Acc: {correct_num_batch/all_num_batch:.4f}\t", flush=True
                )
            log_every = 100
            if iteration % log_every == 0:
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
                val_preprocess = build_transform(n_px=224, is_train=False)
                dataset = dataset_class(
                    train_set=args.train_set,
                    eval_set=args.eval_set,
                    preprocess=val_preprocess,
                    classes_list_path=args.classes_list_path,
                    classes_list_ignore_during_val_path=args.classes_list_ignore_during_val_path,
                    location=args.data_location,
                    batch_size=args.batch_size
                )
                val_loader = dataset.test_loader
                model.eval()
                val_loss = 0.0
                num_samples_val = 0.
                num_samples_old_classes_val = 0.
                correct_samples_val = 0.
                correct_samples_old_classes_val = 0.

                with torch.no_grad():
                    for batch in val_loader:
                        batch = maybe_dictionarize(batch)
                        inputs = batch["images"].cuda()
                        outputs = model(inputs)
                        labels = batch['labels'].cuda()
                        new_class_targets = torch.where(labels == single_trained_class_label, 1, 0).float()
                        loss = criterion(outputs[:, single_trained_class_label], new_class_targets)
                        preds = (torch.sigmoid(outputs[:, single_trained_class_label]) > 0.5).float()
                        correct_num_batch = (preds == new_class_targets).float().sum()

                        probabilities_all_classes = torch.sigmoid(outputs)
                        _, preds_old_classes = torch.max(probabilities_all_classes[:, :single_trained_class_label], 1)
                        old_class_mask = labels != single_trained_class_label
                        old_labels = labels[old_class_mask]
                        preds_old_classes = preds_old_classes[old_class_mask]
                        correct_samples_old_classes_val += (preds_old_classes == old_labels).sum().item()
                        all_num_batch = len(preds)
                        num_samples_val += all_num_batch
                        num_samples_old_classes_val += len(old_labels)
                        correct_samples_val += correct_num_batch
                        val_loss += loss
                val_acc = correct_samples_val/num_samples_val
                vall_acc_old_classes = correct_samples_old_classes_val/num_samples_old_classes_val
                val_loss = val_loss / num_samples_val
                writer.add_scalar('Acc_i/val', val_acc, iteration)
                writer.add_scalar('Acc_old_classes_i/val', vall_acc_old_classes, iteration)
                writer.add_scalar('Loss_i/val', val_loss, iteration)
                print(
                    f"Iteration: {iteration}\t"
                    f"Eval old classes acc: {vall_acc_old_classes:.4f}\t"
                    f"Eval loss: {val_loss:.4f}\t Eval acc: {val_acc:.4f}\t", flush=True
                )

                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    model_path = os.path.join(args.save, f'checkpoint_iteration_{iteration}.pt')
                    print('Saving model to', model_path)
                    utils.torch_save(model, model_path)

            iteration += 1

if __name__ == '__main__':
    args = parse_arguments()
    add_class_into_model(args)
