import os
import json

import torch
import numpy as np

from src.models import utils
from src.datasets.common import maybe_dictionarize, get_dataset
from src.models.utils import LabelSmoothing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import onnxruntime

def eval_single_dataset(image_classifier, dataset, args):
    idx_to_class_name = {}
    with open(args.classes_list_path, "r") as f:
        for index, line in enumerate(f):
            idx_to_class_name[index] = line.rstrip()
    
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()
    if args.ONNX_model is not None:
        ort_session = onnxruntime.InferenceSession(args.ONNX_model, providers=['CPUExecutionProvider'])

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    dataloader = dataset.data_loader
    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        val_gts_onnx = []
        val_preds_onnx = []
        val_scores_onnx = []
        top1, correct, n = 0., 0., 0.
        val_gts = []
        val_preds = []
        val_scores = []
        val_epoch_loss = 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            #----------------ONNX--------------------
            if args.ONNX_model is not None:
                ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy().astype(np.float16)}
                # ort_inputs = {ort_session.get_inputs()[0].name: x.detach().cpu().numpy()}
                logits_onnx = ort_session.run(None, ort_inputs)[0]
                pred_onnx = np.argmax(logits_onnx, axis=1)
                val_gts_onnx.extend(y.detach().cpu().numpy())
                val_preds_onnx.extend(pred_onnx)
                val_scores_onnx.append(logits_onnx)

            #----------------ONNX--------------------


            logits = utils.get_logits(x, model)
            val_scores.append(logits.detach().cpu().numpy())
            loss = loss_fn(logits, y)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

            pred = torch.argmax(logits, dim=1)
            val_preds.extend(pred.detach().cpu().numpy())
            val_gts.extend(y.detach().cpu().numpy())
            val_epoch_loss += loss.item()

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    
    preds_logits = np.concatenate(val_scores, axis=0)
    top_4_acc = top_k_acc(val_gts, preds_logits)

    if args.ONNX_model is not None:
        preds_logits = np.concatenate(val_scores_onnx, axis=0)
        top_4_acc_onnx = top_k_acc(val_gts_onnx, preds_logits)
        val_acc_onnx = accuracy_score(val_gts_onnx, val_preds_onnx)
        print(f"val_acc_onnx in eval.py: {val_acc_onnx:.4f}")
        print(f"val_top_4_acc_onnx in eval.py:{top_4_acc_onnx:.4f}")
    val_acc = accuracy_score(val_gts, val_preds)
    val_precison = precision_score(val_gts, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_gts, val_preds, average='weighted')
    val_f1_score = f1_score(val_gts, val_preds, average='weighted')
    val_loss = val_epoch_loss / len(dataloader)
    metrics["val_top_4_acc"] = top_4_acc
    metrics["val_acc"] = val_acc
    metrics["val_precison"] = val_precison
    metrics["val_recall"] = val_recall
    metrics["val_f1_score"] = val_f1_score
    metrics["val_loss"] = val_loss



    if args.confusion_matrix:
        idx_to_class_name = {}
        with open(args.classes_list_path, "r") as f:
            for index, line in enumerate(f):
                idx_to_class_name[index] = line.rstrip()
        labels = [i for i in range(len(idx_to_class_name))]
        target_names = [idx_to_class_name[idx] for idx in labels]
        cm = confusion_matrix(val_gts, val_preds,labels=labels)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot()
        fig, ax = plt.subplots(figsize=(25, 25))
        cm_display.plot(ax=ax)
        plt.xticks(rotation=90)
        plt.tight_layout()
        save_cm_path = os.path.join(args.save, "confusion_matrix" + ".png")
        plt.savefig(save_cm_path, dpi=300)
        print(classification_report(val_gts, val_preds, target_names=target_names,labels=labels))

    return metrics

def top_k_acc(val_gts, preds_logits):
    top_4_predictions = np.argsort(preds_logits, axis=1)[:, -4:]
    num_samples = len(top_4_predictions)
    num_correct = 0
    for i in range(num_samples):
        if val_gts[i] in top_4_predictions[i]:
            num_correct += 1
    top_4_acc = num_correct / num_samples
    return top_4_acc



def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    dataset_name = args.eval_datasets
    if dataset_name == "SampleFileLoader":
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, "overall_balanced.json")
    print('Evaluating on', args.eval_datasets)
    dataset = get_dataset(args, is_train=False)
    results = eval_single_dataset(image_classifier, dataset, args)

    if 'top1' in results:
        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        info["top1_acc"] = results['top1']
    if 'val_top_4_acc' in results:
        print(f"{dataset_name} Top-4 accuracy: {results['val_top_4_acc']:.4f}")

    for key, val in results.items():
        if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
            print(f"{dataset_name} {key}: {val:.4f}")
        info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info