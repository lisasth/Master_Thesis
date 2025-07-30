import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.modeling import ImageClassifier
from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize, get_dataset
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io
from tqdm import tqdm
from src.models.early_stopping import EarlyStopping
from src.models import utils
from src.datasets.common import TwoHead


INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM = 512, [512], 1


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def train_classification_head_on_backbone(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Determine the label for the new class, new class must be given
    single_trained_class_label = args.single_trained_class_index - 1

    # Load models
    finetuned = ImageClassifier.load(args.load)
    backbone = finetuned.image_encoder.to(device)
    backbone.eval()
    class_head = MLPHead(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM).to(device)

    train_dataset = get_dataset(args, is_train=True)
    train_dataloader = train_dataset.data_loader

    criterion = nn.BCEWithLogitsLoss()
    parameters = class_head.parameters() 
    optimizer = optim.AdamW(parameters, lr=args.lr,
                            weight_decay=args.wd)  # Optimize only the classification head's weights and biases
    writer = SummaryWriter(log_dir=args.logdir)
    iteration = 0

    num_samples_iterations = 0.
    correct_samples_iterations = 0.
    train_loss_iterations = 0.
    print_every = 10

    early_stop = EarlyStopping(save_path=args.save, patience=args.early_stop_patience, verbose=True, min_iteration=args.early_stop_min_iterations)
    for epoch in range(args.epochs):
        if early_stop.early_stop_flag:
            break
        for i, batch in enumerate(train_dataloader):
            backbone.to(device)
            backbone.eval()
            class_head.to(device)
            class_head.train()
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            embeddings = backbone(inputs)
            outputs = class_head(embeddings)
            labels = batch['labels'].cuda()
            one_hot_label = torch.where(labels == single_trained_class_label, 1.0, 0.0).float().unsqueeze(dim=1)
            loss = criterion(outputs, one_hot_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_num_batch = (preds == one_hot_label).float().sum()
            all_num_batch = len(preds)
            num_samples_iterations += all_num_batch
            correct_samples_iterations += correct_num_batch
            train_loss_iterations += loss

            if i % print_every == 0:
                print(f"{one_hot_label.sum()} images from the single_trained_class_label")
                percent_complete = 100 * i / len(train_dataloader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dataset.data_loader)}]\t"
                    f"Loss: {loss.item():.4f}\t Acc: {correct_num_batch / all_num_batch:.4f}\t", flush=True
                )
            log_every = 100
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
                val_dataset = get_dataset(args, is_train=False)
                val_dataloader = val_dataset.data_loader

                class_head.eval()
                val_loss = 0.0
                num_samples_val = 0.
                correct_samples_val = 0.

                with torch.no_grad():
                    for batch in tqdm(val_dataloader):
                        batch = maybe_dictionarize(batch)
                        inputs = batch["images"].cuda()
                        embeddings = backbone(inputs)
                        outputs = class_head(embeddings)

                        labels = batch['labels'].cuda()
                        one_hot_label = torch.where(labels == single_trained_class_label, 1, 0).float().unsqueeze(dim=1)
                        loss = criterion(outputs, one_hot_label)
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct_num_batch = (preds == one_hot_label).float().sum()
                        all_num_batch = len(preds)
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
                    early_stop(val_acc, class_head, iteration)
                    if early_stop.early_stop_flag:
                        print("Early stopping terminates the training, since eval performance does not increase anymore")
                        break

            iteration += 1


if __name__ == '__main__':
    args = parse_arguments()
    fnv_classes = []
    with open(args.classes_list_path, "r") as f:
        for line in f:
            fnv_classes.append(line.rstrip())
    all_model_folder = args.save
    fnv_class = args.single_trained_class
    args.save = os.path.join(args.save, fnv_class)
    args.single_trained_class_index = fnv_classes.index(fnv_class) + 1
    args.train_sample_file_path = os.path.join(args.train_sample_files_folder_path, f"{fnv_class}.json")
    args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, f"{fnv_class}.json")
    train_classification_head_on_backbone(args)
