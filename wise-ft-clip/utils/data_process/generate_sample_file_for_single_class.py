import os
import random
import json
import warnings
import argparse
import math
from generate_sample_file_for_dataset import (
    count_file,
    get_pack_type_to_dataset_name_to_num_files,
    get_priority,
    calculate_images_to_select,
    check_top_down_less_than_cctv, 
    generate_single_class_dataset
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--num_images_each_pack_from_each_class",
        type=int,
        default=265,
    )
    parser.add_argument(
        "--path_to_class_list",
        type=str,
    )
    parser.add_argument(
        "--single_trained_class",
        type=str,
    )
    parser.add_argument(
        "--sample_files_folder_path",
        type=str,
    )

    parsed_args = parser.parse_args()

    return parsed_args
    

if __name__ == "__main__":
    args = parse_arguments()
    root = args.root
    images_needed_each_pack_original = args.num_images_each_pack_from_each_class
    path_to_class_list = args.path_to_class_list
    valid_image_extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    sample_files_folder_path = args.sample_files_folder_path
    os.makedirs(sample_files_folder_path, exist_ok=True)
    print(f"Save sample files in {sample_files_folder_path}")

    class_list = []
    with open(path_to_class_list, "r") as f:
        for line in f:
            class_list.append(line.rstrip())
    
    single_trained_class = args.single_trained_class

    ##### Make sure that we have at least one image per class from each class in the dataset. (If not, it will cause the training to fail)
    if args.num_images_each_pack_from_each_class < len(class_list):
        args.num_images_each_pack_from_each_class = len(class_list)
        print("Warning: we need at least 1 image from each class for the single-head training. "
              "The parameter num_images_each_pack_from_each_class is now automatically reset to the number of classes")
    
    sample_file = generate_single_class_dataset(root, images_needed_each_pack_original, valid_image_extensions, class_list, single_trained_class)

    if set(class_list) != set(sample_file.keys()):
        warnings.warn(f"Classes {set(class_list)-set(sample_file.keys())} missing in sample_file")
    sample_file_json = json.dumps(sample_file, indent=4)
    sample_file_output_path = os.path.join(sample_files_folder_path, f"{single_trained_class}.json")
    with open(sample_file_output_path, "w") as f:
        f.write(sample_file_json)
    print(f"Finished generating sample file for {single_trained_class}")
    print("------------------------------------")