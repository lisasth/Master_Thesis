import os
import random
import copy
import json
import warnings
import argparse
import math
from generate_sample_file_for_dataset import (
    count_file,
    get_pack_type_to_dataset_name_to_num_files,
    get_priority,
    calculate_images_to_select,
    check_top_down_less_than_cctv
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

    sample_file = {}
    for class_name in class_list:
        percentage_each_priority = 0.8
        dataset_name_to_num_files = count_file(root=root, class_name=class_name)
        pack_type_to_dataset_name_to_num_files = get_pack_type_to_dataset_name_to_num_files(dataset_name_to_num_files)
        for pack_name, pack_dict in pack_type_to_dataset_name_to_num_files.items():
            if "Train" in root and pack_name == "no_pack_type":
                images_needed_each_pack = images_needed_each_pack_original * 3
            elif "Train" in root and class_name == "Empty" and pack_name != "no_pack_type":
                continue
            elif class_name in ["Almond", "Cashew", "Pistachio"]:
                images_needed_each_pack = images_needed_each_pack_original * 3 / 2
            else:
                images_needed_each_pack = images_needed_each_pack_original
            pack_dict = get_priority(pack_dict, class_name)
            pack_dict = calculate_images_to_select(pack_dict, images_needed_each_pack, percentage_each_priority)
            check_top_down_less_than_cctv(pack_dict, percentage_each_priority)

            for dataset_name, info in pack_dict.items():
                src_path = os.path.join(root, dataset_name, class_name)
                if "images_to_select" in info:
                    image_files = os.listdir(src_path)
                    selected_files = random.sample(image_files, info["images_to_select"])

                    for image_file in selected_files:
                        file_name_extension = os.path.splitext(image_file)[1]
                        if file_name_extension not in valid_image_extensions:
                            print(f"{os.path.join(src_path, image_file)} is not an image")
                            continue
                        else:
                            image_path = os.path.join(src_path, image_file)
                            if class_name not in sample_file:
                                sample_file[class_name] = [image_path]
                            else:
                                sample_file[class_name].append(image_path)

    if set(class_list) != set(sample_file.keys()):
        warnings.warn(f"Classes {set(class_list)-set(sample_file.keys())} missing in sample_file")
    sample_file_json = json.dumps(sample_file, indent=4)
    sample_file_output_path = os.path.join(sample_files_folder_path, f"overall_balanced.json")
    with open(sample_file_output_path, "w") as f:
        f.write(sample_file_json)
    print(f"Finished generating sample file for overall balanced dataset")
    print("------------------------------------")