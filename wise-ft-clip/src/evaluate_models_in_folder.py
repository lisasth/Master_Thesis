import os

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.camptum_visual import camptum_visual
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from torch.utils.tensorboard import SummaryWriter
import tensorflow_io
import re

torch.multiprocessing.set_sharing_strategy('file_system')


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta

def custom_sort_key(s):
    s = os.path.splitext(s)[0]
    match = re.search(r'(\d+)$', s)
    if match:
        return (0, int(match.group(1)))
    else:
        return (1, s)

def wise_ft(args):
    assert args.save is not None, 'Please provide a path to store models'
    log_dir = os.path.join(args.logdir, "eval_models_tensorboard", args.eval_set)
    writer = SummaryWriter(log_dir = log_dir)
    for model_to_evaluate in sorted(os.listdir(args.models_folder), key=custom_sort_key):
        if model_to_evaluate.endswith(".pt"):
            print(f"-----------------------------------")
            print(f"Evaluating {model_to_evaluate}")
            iteration = os.path.splitext(model_to_evaluate)[0].split("_")[-1]
            finetuned_checkpoint = os.path.join(args.models_folder, model_to_evaluate)
            assert len(args.load) == 2
            zeroshot_checkpoint, _ = args.load

            # Load models
            zeroshot = ImageClassifier.load(zeroshot_checkpoint)
            finetuned = ImageClassifier.load(finetuned_checkpoint)
            theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
            theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
            del zeroshot

            if args.fisher is None:
                fishers = None
            else:
                fisher_0_file, fisher_1_file = args.fisher
                fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
                fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
                fishers = fisher_0, fisher_1

            # make sure checkpoints are compatible
            assert set(theta_0.keys()) == set(theta_1.keys())

            alpha = args.alpha[0]
            theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

            # update the model (in-place) acccording to the new weights
            finetuned.load_state_dict(theta)

            # save model
            finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

            # camptum visual
            if args.camptum_visual:
                camptum_visual(finetuned, args)

            # evaluate
            eval_results = evaluate(finetuned, args)
            for key in eval_results.keys():
                if "val_top_4_acc" in key:
                    dataset_name = key.split(":")[0].strip() 
                    writer.add_scalar(f'Acc_top4_i/val_{dataset_name}', eval_results[key], iteration)
                if "val_acc" in key:
                    dataset_name = key.split(":")[0].strip() 
                    writer.add_scalar(f'Acc_i/val_{dataset_name}', eval_results[key], iteration)
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


if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)
