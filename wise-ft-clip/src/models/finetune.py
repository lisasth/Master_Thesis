import os
import copy
import time
import torch
from src.args import parse_arguments
from src.datasets.common import maybe_dictionarize, get_dataset
from src.models.eval import evaluate
from src.models.modeling import ImageClassifier, ContrastiveImageClassifier
from src.models.utils import cosine_lr, LabelSmoothing, mixup_data, mixup_data_different_classes
import src.models.utils as utils
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.models.contrastive_learn import SupConLoss
import json
from src.models.early_stopping import EarlyStopping

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#dev = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")

def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    image_classifier = ImageClassifier.load(args.load)
    if args.contrastive:
        image_classifier = ContrastiveImageClassifier(
            image_classifier.image_encoder,
            image_classifier.classification_head,
            image_classifier.process_images,
            contrastive_output_size=128
        )
    model = image_classifier
    image_classifier.process_images = True
    print_every = 10
    if args.train_dataset == "SampleFileLoader":
        args.train_sample_file_path = os.path.join(args.train_sample_files_folder_path, "overall_balanced.json")
    train_dataset = get_dataset(args=args, is_train=True)
    num_batches = len(train_dataset.data_loader)

    model = model.to(dev)
    devices = list(range(torch.cuda.device_count()))
    #if torch.backends.mps.is_available() and torch.backends.mps.is_built():
      #  print("using mps")
    model = torch.nn.DataParallel(model, device_ids=devices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    model.train()

    if args.freeze_classification_head:
        for name, param in model.named_parameters():
            if "classification_head" in name:
                param.requires_grad = False

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classification_head" not in name:
                param.requires_grad = False

    classes_list = []
    with open(args.classes_list_path, "r") as f:
        for line in f:
            classes_list.append(line.strip())
    num_classes = len(classes_list)
    if args.ls > 0:
        loss_func_entropy = LabelSmoothing(args.ls)
    elif args.entropy_loss_name == "CrossEntropyLoss":
        entropy_weights = torch.ones(num_classes)
        if args.class_entropy_loss_weights_path:
            with open(args.class_entropy_loss_weights_path) as f:
                class_entropy_weights = json.load(f)
            for class_name, weight in class_entropy_weights.items():
                class_index = classes_list.index(class_name)
                entropy_weights[class_index] = weight
        loss_func_entropy = torch.nn.CrossEntropyLoss(weight=entropy_weights)#.to("mps")
    elif args.entropy_loss_name == "BCEWithLogitsLoss":
        loss_func_entropy = torch.nn.BCEWithLogitsLoss()
    else:
        raise Exception("Unsupported entropy loss, please choose between 'CrossEntropyLoss' and 'BCEWithLogitsLoss'")

    if args.contrastive:
        loss_func_contra = SupConLoss(temperature=args.temperature)
        print("Train with contrastive learning")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    writer = SummaryWriter(log_dir=args.logdir)
    iteration = 0
    early_stop = EarlyStopping(save_path=args.save, patience=args.early_stop_patience, verbose=True,
                               min_iteration=args.early_stop_min_iterations)
    for epoch in range(args.epochs):
        if early_stop.early_stop_flag:
            break
        train_gts = []
        train_preds = []
        train_epoch_loss = 0.
        train_entropy_epoch_loss = 0.
        train_contra_epoch_loss = 0.
        train_gts_iterations = []
        train_preds_iterations = []
        train_loss_iterations = 0.
        train_entropy_loss_iterations = 0.
        train_contra_loss_iterations = 0.
        model.train()
        train_dataloader = train_dataset.data_loader

        for i, batch in enumerate(train_dataloader):

            start_time = time.time()

            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)

            if args.contrastive:
                inputs = batch['images']
                labels = batch['labels'].to(dev)

                if args.mixup:
                    mixup_start_time = time.time()
                    inputs[0], mixed_label_1 = mixup_data_different_classes(inputs[0].to(dev), labels)
                    inputs[1], mixed_label_2 = mixup_data_different_classes(inputs[1].to(dev), labels)
                    print(mixed_label_1)

                    print(f"mixup took {time.time() - mixup_start_time}")

                images = torch.cat([inputs[0], inputs[1]], dim=0).to(dev)
                data_time = time.time() - start_time
                batch_size = labels.shape[0]
                pred_supcon, logits = model(images)
                f1, f2 = torch.split(pred_supcon, [batch_size, batch_size], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_contra = loss_func_contra(features, labels)
                labels = torch.cat((labels, labels))
                if isinstance(loss_func_entropy, torch.nn.BCEWithLogitsLoss):
                    one_hot_labels = utils.get_one_hot_labels(copy.deepcopy(labels), num_classes, device)
                    loss_classify = loss_func_entropy(logits, one_hot_labels)
                elif isinstance(loss_func_entropy, torch.nn.CrossEntropyLoss):
                    loss_classify = loss_func_entropy(logits, labels)
                loss = args.contra_loss_weight * loss_contra + (1 - args.contra_loss_weight) * loss_classify

                train_entropy_epoch_loss += loss_classify.item()
                train_entropy_loss_iterations += loss_classify.item()
                train_contra_epoch_loss += loss_contra.item()
                train_contra_loss_iterations += loss_contra.item()
                # image_to_save = inputs[0][0].detach().cpu()
            else:
                inputs = batch['images'].to(dev)
                labels = batch['labels'].to(dev)
                if args.mixup:
                    inputs = mixup_data(inputs, labels)
                data_time = time.time() - start_time

                logits = model(inputs)

                loss = loss_func_entropy(logits, labels)
                # image_to_save = inputs[0].detach().cpu()

            # tranformed_img_save_folder = os.path.join("saved_train_images", args.save)
            # os.makedirs(tranformed_img_save_folder, exist_ok=True)
            # tranformed_img_save_path = os.path.join(tranformed_img_save_folder, f"transformed_img_batch_{i}_epoch_{epoch}.png")

            # save_image(image_to_save, tranformed_img_save_path)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            pred = torch.argmax(logits, dim=1)
            train_preds.extend(pred.detach().cpu())
            train_gts.extend(labels.detach().cpu())
            train_preds_iterations.extend(pred.detach().cpu())
            train_gts_iterations.extend(labels.detach().cpu())
            train_epoch_loss += loss.item()
            train_loss_iterations += loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / len(train_dataloader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dataset.data_loader)}]\t"
                    f"Loss: {loss.item():.6f}\t Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t", flush=True
                )

            log_every = 200
            if iteration % log_every == 0:

                train_acc = accuracy_score(train_gts_iterations, train_preds_iterations)
                train_precison = precision_score(train_gts_iterations, train_preds_iterations, average='weighted',
                                                 zero_division=0)
                train_recall = recall_score(train_gts_iterations, train_preds_iterations, average='weighted')
                train_f1_score = f1_score(train_gts_iterations, train_preds_iterations, average='weighted')
                train_loss = train_loss_iterations / len(train_gts_iterations)
                if args.contrastive:
                    train_entropy_loss = train_entropy_loss_iterations / len(train_gts_iterations)
                    train_contra_loss = train_contra_loss_iterations / len(train_gts_iterations)
                    writer.add_scalar('Entropy_loss_i/train', train_entropy_loss, iteration)
                    writer.add_scalar('Contra_loss_i/train', train_contra_loss, iteration)
                writer.add_scalar('Acc_i/train', train_acc, iteration)
                writer.add_scalar('Precision_i/train', train_precison, iteration)
                writer.add_scalar('Recall_i/train', train_recall, iteration)
                writer.add_scalar('F1-score_i/train', train_f1_score, iteration)
                writer.add_scalar('Loss_i/train', train_loss, iteration)
                train_gts_iterations = []
                train_preds_iterations = []
                train_loss_iterations = 0.
                train_entropy_loss_iterations = 0.
                train_contra_loss_iterations = 0.

                if args.freeze_encoder:
                    image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
                else:
                    image_classifier = model.module

                if args.contrastive:
                    image_classifier = ImageClassifier(model.module.image_encoder, model.module.classification_head)

                eval_results = evaluate(image_classifier, args)
                for key in eval_results.keys():
                    if "val_acc" in key:
                        dataset_name = key.split(":")[0].strip()
                        writer.add_scalar(f'Acc_i/val_{dataset_name}', eval_results[key], iteration)
                        val_acc = eval_results[key]
                    if "val_precison" in key:
                        dataset_name = key.split(":")[0].strip()
                        writer.add_scalar(f'Precision_i/val_{dataset_name}', eval_results[key], iteration)
                    if "val_recall" in key:
                        dataset_name = key.split(":")[0].strip()
                        writer.add_scalar(f'Recall_i/val_{dataset_name}', eval_results[key], iteration)
                    if "val_f1_score" in key:
                        dataset_name = key.split(":")[0].strip()
                        writer.add_scalar(f'F1-score_i/val_{dataset_name}', eval_results[key], iteration)
                    if "val_loss" in key:
                        dataset_name = key.split(":")[0].strip()
                        writer.add_scalar(f'Loss_i/val_{dataset_name}', eval_results[key], iteration)
                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                    early_stop(val_acc, image_classifier, iteration)
                    if early_stop.early_stop_flag:
                        print("Early stopping")
                        break
                    model = model.to(dev)

            iteration += 1

        train_acc = accuracy_score(train_gts, train_preds)
        train_precison = precision_score(train_gts, train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(train_gts, train_preds, average='weighted')
        train_f1_score = f1_score(train_gts, train_preds, average='weighted')
        train_loss = train_epoch_loss / len(train_dataloader)
        if args.contrastive:
            train_entropy_loss = train_entropy_epoch_loss / len(train_dataloader)
            train_contra_loss = train_contra_epoch_loss / len(train_dataloader)
            writer.add_scalar('Entropy_loss/train', train_entropy_loss, iteration)
            writer.add_scalar('Contra_loss/train', train_contra_loss, iteration)
        writer.add_scalar('Acc/train', train_acc, epoch + 1)
        writer.add_scalar('Precision/train', train_precison, epoch + 1)
        writer.add_scalar('Recall/train', train_recall, epoch + 1)
        writer.add_scalar('F1-score/train', train_f1_score, epoch + 1)
        writer.add_scalar('Loss/train', train_loss, epoch + 1)

        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        if args.contrastive:
            image_classifier = ImageClassifier(model.module.image_encoder, model.module.classification_head)

        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch + 1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            # optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            # torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)
        for key in eval_results.keys():
            if "val_acc" in key:
                dataset_name = key.split(":")[0].strip()
                writer.add_scalar(f'Acc/val_{dataset_name}', eval_results[key], epoch + 1)
            if "val_precison" in key:
                dataset_name = key.split(":")[0].strip()
                writer.add_scalar(f'Precision/val_{dataset_name}', eval_results[key], epoch + 1)
            if "val_recall" in key:
                dataset_name = key.split(":")[0].strip()
                writer.add_scalar(f'Recall/val_{dataset_name}', eval_results[key], epoch + 1)
            if "val_f1_score" in key:
                dataset_name = key.split(":")[0].strip()
                writer.add_scalar(f'F1-score/val_{dataset_name}', eval_results[key], epoch + 1)
            if "val_loss" in key:
                dataset_name = key.split(":")[0].strip()
                writer.add_scalar(f'Loss/val_{dataset_name}', eval_results[key], epoch + 1)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
