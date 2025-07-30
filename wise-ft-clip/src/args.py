import argparse
import os

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=(
            'Interpolation coefficient for ensembling. '
            'Users should specify N-1 values, where N is the number of '
            'models being ensembled. The specified numbers should sum to '
            'less than 1. Note that the order of these values matter, and '
            'should be the same as the order of the classifiers being ensembled.'
        )
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )
    parser.add_argument(
        "--classes_list_path",
        type=str,
        default=None,
        help="Path to the list of class names to be included in the dataset",
    )
    parser.add_argument(
        "--classes_list_ignore_during_val_path",
        type=str,
        default=None,
        help="Path to the list of class names to be ignored in the dataset",
    )
    parser.add_argument(
        "--camptum_visual",
        default=False,
        action="store_true",
        help="Flag for camptum visualization",
    )

    parser.add_argument(
        "--confusion_matrix",
        default=False,
        action="store_true",
        help="Flag for generating confusion matrix",
    )

    parser.add_argument(
        "--ONNX_model",
        type=str,
        default=None,
        help="Path to the ONNX model to be evaluated",
    )
    parser.add_argument(
        "--classnames_to_encode",
        type=str,
        default=None,
        help="Path to the list of class names to be encoded by text encoder",
    )
    parser.add_argument(
        "--freeze_classification_head",
        default=False,
        action="store_true",
        help="Flag for classification head during finetuning",
    )
    parser.add_argument(
        "--freeze_backbone",
        default=False,
        action="store_true",
        help="Flag for freezing backbone during finetuning",
    )
    parser.add_argument(
        "--contrastive",
        default=False,
        action="store_true",
        help="Flag for contrastive learning",
    )
    parser.add_argument(
        "--mixup",
        default=False,
        action="store_true",
        help="Flag for mixup data within the same class",
    )
    parser.add_argument(
        "--weighted_sampler",
        default=False,
        action="store_true",
        help="Flag for using WeightedRandomSampler during training",
    )
    parser.add_argument(
        "--class_scanned_percentage_path",
        type=str,
        default=None,
        help="Path to json file with labels(numbers) as keys, percentages as values",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        default=None,
        help="Folder name of the training set",
    )
    parser.add_argument(
        "--eval_set",
        type=str,
        default=None,
        help="Folder name of the evaluation set",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature value for contrasitve loss, default value 0.07",
    )
    parser.add_argument(
        "--contra_loss_weight",
        type=float,
        default=0.5,
        help="Weight for contrasitve loss, default is 0.5",
    )
    parser.add_argument(
        "--models_folder",
        type=str,
        default=None,
        help="Folder name of models to be evaluated",
    )
    parser.add_argument(
        "--class_entropy_loss_weights_path",
        type=str,
        default=None,
        help="Path to json file with labels(numbers) as keys, entropy loss weights as values",
    )
    parser.add_argument(
        "--single_trained_class_index",
        type=int,
        default=None,
        help="The index of the new class in the class name list (the index starting from 1 instead of 0 here)",
    )
    parser.add_argument(
        "--add_new_class",
        default=False,
        action="store_true",
        help="Flag for adding a new class into the previous model",
    )
    parser.add_argument(
        "--entropy_loss_name",
        type=str,
        default="CrossEntropyLoss",
        help="Entropy loss name, choose between 'CrossEntropyLoss' and 'BCEWithLogitsLoss'",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="This number refers to the repeated times when the evaluation performance does not increase, the early stopping will stop the training."

    )
    parser.add_argument(
        "--early_stop_min_iterations",
        type=int,
        default=3000,
        help="This number refers to the minimun iterartions, when the early stopping starts to save the model with best evaluation performance."

    )
    parser.add_argument(
        "--train_sample_file_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_sample_file_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_sample_files_folder_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_sample_files_folder_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--class_heads_folder_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--single_trained_class",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Directory to log metrics",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clip_wise_ft",
        help="Model type to train",
    )

    parser.add_argument(
        "--log_frequency",
        type=int,
        default=200,
        help="Number of iterations between two logging steps"
    )
    parser.add_argument(
        "--hidden_dim",
        default=64,
        type=int,
        help=(
            'Hidden dimentions for a classification head.'
        )
    )
    parser.add_argument(
        "--path_to_full_class_list",
        default = None,
        type=str,
    )
    parser.add_argument(
        "--path_to_class_list_to_train",
        type=str,
        default = None
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
