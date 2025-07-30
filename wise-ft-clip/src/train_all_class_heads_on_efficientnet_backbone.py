import os
from src.args import parse_arguments

from src.train_classification_single_head_on_efficientnet_backbone import train_classification_head_on_efficientnet_backbone

if __name__ == '__main__':
    args = parse_arguments()
    fnv_classes = []
    with open(args.classes_list_path, "r") as f:
        for line in f:
            fnv_classes.append(line.rstrip())
    all_model_folder = args.save
    for i, fnv_class in enumerate(fnv_classes):
        print("-------------------------------------------------------------")
        print(f"Start training classification head for {fnv_class}")
        args.single_trained_class_index = i + 1
        args.save = os.path.join(all_model_folder, fnv_class)
        args.train_sample_file_path = os.path.join(args.train_sample_files_folder_path, f"{fnv_class}.json")
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, f"{fnv_class}.json")
        print(f"Save path is {args.save}")
        train_classification_head_on_efficientnet_backbone(args)
        print(f"Finished training classification head for {fnv_class}")