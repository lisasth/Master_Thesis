import os
from src.args import parse_arguments

from src.train_classification_single_head_on_backbone import train_classification_head_on_backbone
from utils.data_process.generate_sample_file_for_dataset import read_class_list

if __name__ == '__main__':
    args = parse_arguments()
     ##TODO: we should exclude the classes_list_path argument entirely in the future. Now this is for make sure  we have backwards compatibility to argo-workflows for now
    if args.classes_list_path is None:
        args.classes_list_path = args.path_to_full_class_list
    # if args.path_to_full_class_list is None:
    #     args.path_to_full_class_list = args.classes_list_path
    # fnv_classes = read_class_list(args.path_to_full_class_list)
    classes_to_train = read_class_list(args.path_to_class_list_to_train)
    
    all_model_folder = args.save
    for i, fnv_class in enumerate(classes_to_train):
        print("-------------------------------------------------------------")
        print(f"Start training classification head for {fnv_class}")
        args.single_trained_class_index = i + 1
        args.save = os.path.join(all_model_folder, fnv_class)
        args.train_sample_file_path = os.path.join(args.train_sample_files_folder_path, f"{fnv_class}.json")
        args.eval_sample_file_path = os.path.join(args.eval_sample_files_folder_path, f"{fnv_class}.json")
        print(f"Save path is {args.save}")
        
        train_classification_head_on_backbone(args)
        print(f"Finished training classification head for {fnv_class}")