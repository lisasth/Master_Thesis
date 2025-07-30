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
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
import src.datasets as datasets
from torchvision.transforms import transforms
from PIL import Image
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io

LEAST_ITERATION = 1
MIN_PERCENTAGE_WRONG_PREDS = 0.1


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


def large_logits_penalty(logits, threshold=5.0, lambda_penalty=0.01):
    penalty_loss = torch.clamp(logits.abs() - threshold, min=0.0)
    non_zero_penalty_loss = penalty_loss[penalty_loss > 0]
    if len(non_zero_penalty_loss) > 0:
        non_zero_penalty_loss = non_zero_penalty_loss.mean() * lambda_penalty
    else:
        non_zero_penalty_loss = 0.0
    return non_zero_penalty_loss


def add_class_into_model(args):
    # Determine the label for the new class, new class must be given
    single_trained_class_label = args.single_trained_class_index - 1

    # Load models
    finetuned = ImageClassifier.load(args.load)
    image_encoder = finetuned.image_encoder
    classification_head = finetuned.classification_head
    if args.add_new_class:
        classification_head = ExtendedLinearClassifier(old_classifier=classification_head)
    new_image_classifier = ImageClassifier(image_encoder=image_encoder, classification_head=classification_head)

    # Freeze the backbone
    for param in new_image_classifier.image_encoder.parameters():
        param.requires_grad = False  

    train_preprocess_fn = build_transform(n_px=224, is_train=True)
    dataset_class = getattr(datasets, args.dataset)
    dataset = dataset_class(
        train_set=args.train_set,
        eval_set=args.eval_set,
        preprocess=train_preprocess_fn,
        classes_list_path=args.classes_list_path,
        location=args.data_location,
        batch_size=args.batch_size,
        weighted_sampler=args.weighted_sampler,
        single_trained_class_label=single_trained_class_label,
    )

    criterion = nn.BCEWithLogitsLoss()
    parameters = new_image_classifier.classification_head.parameters()
    
    # Optimize only the new classifier's weights and biases
    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)  
    log_dir = os.path.join(args.save, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    iteration = 0

    num_samples_iterations = 0.
    correct_samples_iterations = 0.
    train_loss_iterations = 0.
    print_every = 1

    new_image_classifier.eval()
    data_loader = dataset.train_loader
    for epoch in range(args.epochs):
        for i, batch in enumerate(data_loader):
            classes_to_train = [single_trained_class_label]
            wrong_classes_to_train = []
            losses_all_trained_classes = []
            new_image_classifier = new_image_classifier.cuda()
            # Set only classification_head to train to make sure the backbone will not change
            new_image_classifier.classification_head.train()
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            outputs = new_image_classifier(inputs)
            labels = batch['labels'].cuda()
            new_class_targets = torch.where(labels == single_trained_class_label, 1.0, 0.0).float()

            # Dynamically find out most confusing classes to the new class and then train the model with those classes
            if iteration > LEAST_ITERATION:
                outputs_single_trained_class = outputs[:, single_trained_class_label]
                outputs_single_trained_class = outputs_single_trained_class.unsqueeze(1).expand_as(outputs)
                wrong_preds = (outputs > outputs_single_trained_class) * new_class_targets.unsqueeze(1)
                _, wrong_preds_classes = wrong_preds.nonzero(as_tuple=True)
                unique_wrong_preds_classes, wrong_preds_counts = wrong_preds_classes.unique(return_counts=True)
                wrong_classes_more_than_th = unique_wrong_preds_classes[
                    wrong_preds_counts > MIN_PERCENTAGE_WRONG_PREDS * args.batch_size]
                wrong_classes_to_train.extend(wrong_classes_more_than_th.tolist())
            
            classes_to_train.extend(wrong_classes_to_train)

            loss = 0.
            for class_to_train in classes_to_train:
                # Modify the label to be 0 if not the class to train, to be 1 if it's the class to train
                one_hot_label = torch.where(labels == class_to_train, 1.0, 0.0).float()
                loss_class_to_train = criterion(outputs[:, class_to_train], one_hot_label)
                losses_all_trained_classes.append(loss_class_to_train.item())
                loss_large_logits_penalty = large_logits_penalty(outputs[:, class_to_train], threshold=1.0,
                                                                 lambda_penalty=0)
                loss += loss_class_to_train + loss_large_logits_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct_num_batch = ((preds == labels).float() * new_class_targets).sum()
            all_num_batch = new_class_targets.sum()
            num_samples_iterations += all_num_batch
            correct_samples_iterations += correct_num_batch
            train_loss_iterations += loss

            if i % print_every == 0:
                print(f"{all_num_batch} images from the single_trained_class_label")
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.4f}\t Acc: {correct_num_batch / all_num_batch:.4f}\t", flush=True
                )
            log_every = 5
            if iteration % log_every == 0:
                train_acc = correct_samples_iterations / num_samples_iterations
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
                val_dataset = dataset_class(
                    train_set=args.train_set,
                    eval_set=args.eval_set,
                    preprocess=val_preprocess,
                    classes_list_path=args.classes_list_path,
                    classes_list_ignore_during_val_path=args.classes_list_ignore_during_val_path,
                    location=args.data_location,
                    batch_size=args.batch_size
                )
                val_loader = val_dataset.test_loader
                new_image_classifier.eval()
                val_loss = 0.0
                num_samples_val = 0.
                correct_samples_val = 0.

                with torch.no_grad():
                    for batch in val_loader:
                        batch = maybe_dictionarize(batch)
                        inputs = batch["images"].cuda()
                        outputs = new_image_classifier(inputs)

                        # Check logits for trained clasees
                        logits_trained_classes = []
                        for class_to_train in classes_to_train:
                            logit_trained_class = outputs[:, class_to_train]
                            logits_trained_classes.append(logit_trained_class)

                        labels = batch['labels'].cuda()
                        new_class_targets = torch.where(labels == single_trained_class_label, 1, 0).float()
                        loss = criterion(outputs[:, single_trained_class_label], new_class_targets)
                        preds = torch.argmax(outputs, dim=1)
                        correct_num_batch = ((preds == labels).float() * new_class_targets).sum()
                        all_num_batch = new_class_targets.sum()
                        num_samples_val += all_num_batch
                        correct_samples_val += correct_num_batch
                        val_loss += loss
                val_acc = correct_samples_val / num_samples_val
                val_loss = val_loss / num_samples_val
                writer.add_scalar('Acc_i/val', val_acc, iteration)
                writer.add_scalar('Loss_i/val', val_loss, iteration)
                print(
                    f"Iteration: {iteration}\t"
                    f"Eval loss: {val_loss:.4f}\t Eval acc: {val_acc:.4f}\t", flush=True
                )

                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    model_path = os.path.join(args.save, f'checkpoint_iteration_{iteration}.pt')
                    print('Saving model to', model_path)
                    new_image_classifier.save(model_path)

            iteration += 1


if __name__ == '__main__':
    args = parse_arguments()
    add_class_into_model(args)
