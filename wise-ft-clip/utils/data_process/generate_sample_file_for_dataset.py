import os
import random
import copy
import json
import warnings
import argparse
import math


def count_file(root, class_name):
    dataset_name_to_num_files = {}
    for dataset_folder in os.scandir(root):
        if dataset_folder.is_dir():
            for subfolder in os.scandir(dataset_folder.path):
                if subfolder.is_dir() and subfolder.name == class_name:
                    files = next(os.walk(subfolder.path))[2]
                    num_files = len(files)
                    dataset_name_to_num_files[dataset_folder.name] = num_files
    return dataset_name_to_num_files


def get_pack_type_to_dataset_name_to_num_files(dataset_name_to_num_files):
    pack_type_to_dataset_name_to_num_files = {}
    for dataset_name in dataset_name_to_num_files.keys():
        no_pack_type = True
        for pack_type in ["loose", "bag", "net"]:
            if pack_type in dataset_name:
                no_pack_type = False
                if pack_type not in pack_type_to_dataset_name_to_num_files.keys():
                    pack_type_to_dataset_name_to_num_files[pack_type] = {}
                pack_type_to_dataset_name_to_num_files[pack_type][dataset_name] = dataset_name_to_num_files[dataset_name]

        if no_pack_type:
            if "no_pack_type" not in pack_type_to_dataset_name_to_num_files.keys():
                pack_type_to_dataset_name_to_num_files["no_pack_type"] = {}
            pack_type_to_dataset_name_to_num_files["no_pack_type"][dataset_name] = dataset_name_to_num_files[dataset_name]
    return pack_type_to_dataset_name_to_num_files


def get_priority(num_dict, class_name):
    num_dict_with_priority = {}
    for dataset_name, num in num_dict.items():
        dataset_name_lower = dataset_name.lower()
        priority = None
        if class_name == "Packaged Product":
            if "cz" in dataset_name_lower: # Data in cz stores
                priority = 5
            elif ("2024_07" in dataset_name_lower
                  or "2024_08" in dataset_name_lower
                  or "2024_09" in dataset_name_lower
                  or "2024_10" in dataset_name_lower) and "sco" in dataset_name_lower: # Data in uk stores
                priority = 5
            elif "top_down_uk" in dataset_name_lower: # Data in uk stores, from top down camera
                priority = 5
            elif "uk-cleaned" in dataset_name_lower: # Data in uk, manually collected (not from stores)
                priority = 0
            elif "top_down" in dataset_name_lower and "uk" not in dataset_name_lower: # Data in Germany, from top down camera, manually collected (not from stores)
                priority = 5
            else:
                priority = 0 # Data in Germany, manually collected (not from stores)
        else:
            if "cz" in dataset_name_lower: # Data in cz stores
                priority = 0
            elif ("2024_07" in dataset_name_lower
                  or "2024_08" in dataset_name_lower
                  or "2024_09" in dataset_name_lower
                  or "2024_10" in dataset_name_lower) and "sco" in dataset_name_lower: # Data in uk stores
                priority = 1
            elif "top_down_uk" in dataset_name_lower: # Data in uk stores, from top down camera
                priority = 2
            elif "uk-cleaned" in dataset_name_lower: # Data in uk, manually collected (not from stores)
                priority = 3
            elif "top_down" in dataset_name_lower and "uk" not in dataset_name_lower: # Data in Germany, from top down camera, manually collected (not from stores)
                priority = 5
            else:
                priority = 4 # Data in Germany, manually collected (not from stores)
        num_dict_with_priority[dataset_name] = {"number of images": num, "priority": priority}
    return num_dict_with_priority


def get_priority_map(datasets):
    """
    Organize datasets by priority.
    """
    priority_map = {}
    for name, info in datasets.items():
        priority = info["priority"]
        if priority not in priority_map:
            priority_map[priority] = []
        priority_map[priority].append((name, info["number of images"]))
    return priority_map


def calculate_images_to_select(datasets_dict, total_images_needed, percentage=0.8):
    """
    Calculate the number of images to select for each dataset based on priority and weights.
    Update the datasets dictionary with the number of images to select.
    """
    priority_map = get_priority_map(datasets_dict)
    sorted_priorities = sorted(priority_map.keys())
    remaining_images_needed = total_images_needed
    finished = False
    for priority in sorted_priorities:
        if finished:
            break
        priority_datasets = priority_map[priority]
        images_to_select = math.ceil(remaining_images_needed * percentage)

        if priority == sorted_priorities[-1]:  # If it's the last priority, take all remaining
            images_to_select = remaining_images_needed

        total_images_in_priority = sum(datasets_dict[name]["number of images"] for name, _ in priority_datasets)

        for dataset_name, num_images in priority_datasets:
            weight = datasets_dict[dataset_name]["number of images"] / total_images_in_priority
            images_for_dataset = math.ceil(images_to_select * weight)

            # Adjust if the calculated number exceeds available images
            if images_for_dataset > datasets_dict[dataset_name]["number of images"]:
                images_for_dataset = datasets_dict[dataset_name]["number of images"]
            datasets_dict[dataset_name]["images_to_select"] = images_for_dataset
            remaining_images_needed -= images_for_dataset
            if remaining_images_needed <= 0:
                finished = True
                break
    return datasets_dict


def check_top_down_less_than_cctv(datasets_dict, percentage_each_priority):
    top_down_select_num = 0
    top_down_total_num = 0
    cctv_mirror_select_num = 0
    cctv_mirror_total_num = 0
    for dataset_name, info in datasets_dict.items():
        if "top_down" in dataset_name:
            top_down_select_num += info.get("images_to_select",0)
            top_down_total_num += info["number of images"]
        else:
            cctv_mirror_select_num += info.get("images_to_select",0)
            cctv_mirror_total_num += info["number of images"]
    if top_down_select_num > cctv_mirror_select_num:
        print("top_down_select_num is more than cctv_mirror_select_num")
        average = (top_down_select_num + cctv_mirror_select_num) / 2
        if cctv_mirror_total_num < average + 2:
            print(f"cctv_mirror_total_num {cctv_mirror_total_num} < average+2 ({average+2}).")
            print("cctv_mirror more than top_down is not possible")
        else:
            cctv_mirror_to_select = average + 2
            top_down_to_select = average - 2
            datasets_dict_cctv_mirror = {}
            datasets_dict_top_down = {}
            for k in datasets_dict.keys():
                if "top_down" not in k:
                    datasets_dict_cctv_mirror[k] = datasets_dict[k]
                else:
                    datasets_dict_top_down[k] = datasets_dict[k]
            datasets_dict_cctv_mirror = calculate_images_to_select(datasets_dict_cctv_mirror, cctv_mirror_to_select, percentage_each_priority)
            datasets_dict_top_down = calculate_images_to_select(datasets_dict_top_down, top_down_to_select, percentage_each_priority)
            datasets_dict_updated = copy.deepcopy(datasets_dict_cctv_mirror)
            datasets_dict_updated.update(datasets_dict_top_down)
            datasets_dict = datasets_dict_updated
            print("Now top_down_select_num should be less than cctv_mirror_select_num")
    return datasets_dict

def generate_single_class_dataset(root, images_needed_each_pack_original, valid_image_extensions, full_class_list, single_trained_class):
    sample_file = {}
    for class_name in full_class_list:
        percentage_each_priority = 0.8
        dataset_name_to_num_files = count_file(root=root, class_name=class_name)
        pack_type_to_dataset_name_to_num_files = get_pack_type_to_dataset_name_to_num_files(dataset_name_to_num_files)
        for pack_name, pack_dict in pack_type_to_dataset_name_to_num_files.items():
            if class_name == single_trained_class:
                images_needed_each_pack = images_needed_each_pack_original
            else:
                images_needed_each_pack = math.ceil(images_needed_each_pack_original/(len(full_class_list)-1))
            if "Train" in root and pack_name == "no_pack_type":
                images_needed_each_pack = images_needed_each_pack * 3
            if "Train" in root and class_name == "Empty" and pack_name != "no_pack_type":
                continue
            if class_name in ["Almond", "Cashew", "Pistachio"]:
                images_needed_each_pack = images_needed_each_pack * 3 / 2
            pack_dict = get_priority(pack_dict, class_name)
            pack_dict = calculate_images_to_select(pack_dict, images_needed_each_pack, percentage_each_priority)
            check_top_down_less_than_cctv(pack_dict, percentage_each_priority)

            for dataset_name, info in pack_dict.items():
                src_path = os.path.join(root, dataset_name, class_name)
                if "images_to_select" in info:
                    image_files = os.listdir(src_path)
                    selected_files = random.sample(image_files, min(max(info["images_to_select"],0),len(image_files)))

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
    return sample_file

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
        "--path_to_full_class_list",
        type=str,
    )
    parser.add_argument(
        "--path_to_class_list_to_train",
        type=str,
        default = None
    )

    parser.add_argument(
        "--sample_files_folder_path",
        type=str,
    )

    parsed_args = parser.parse_args()

    return parsed_args

def read_class_list(path_to_class_list):
    class_list = []
    with open(path_to_class_list, "r") as f:
        for line in f:
            class_list.append(line.rstrip())
    return class_list


if __name__ == "__main__":
    args = parse_arguments()
    root = args.root
    images_needed_each_pack_original = args.num_images_each_pack_from_each_class
    path_to_full_class_list = args.path_to_full_class_list
    path_to_class_list_to_train = args.path_to_class_list_to_train
    valid_image_extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    sample_files_folder_path = args.sample_files_folder_path
    os.makedirs(sample_files_folder_path, exist_ok=True)
    print(f"Save sample files in {sample_files_folder_path}")

    full_class_list = read_class_list(path_to_full_class_list)
    class_list_to_train = read_class_list(path_to_class_list_to_train) if path_to_class_list_to_train is not None else full_class_list

    ##### Make sure that we have at least one image per class from each class in the dataset. (If not, it will cause the training to fail)
    if args.num_images_each_pack_from_each_class < len(full_class_list):
        args.num_images_each_pack_from_each_class = len(full_class_list)
        print("Warning: we need at least 1 image from each class for the single-head training. "
              "The parameter num_images_each_pack_from_each_class is now automatically reset to the number of classes")

    for single_trained_class in class_list_to_train:
        sample_file = generate_single_class_dataset(root, images_needed_each_pack_original, valid_image_extensions, full_class_list, single_trained_class)

        if set(full_class_list) != set(sample_file.keys()):
            warnings.warn(f"Classes {set(full_class_list)-set(sample_file.keys())} missing in sample_file")
        sample_file_json = json.dumps(sample_file, indent=4)
        sample_file_output_path = os.path.join(sample_files_folder_path, f"{single_trained_class}.json")
        with open(sample_file_output_path, "w") as f:
            f.write(sample_file_json)
        print(f"Finished generating sample file for {single_trained_class}")
        print("------------------------------------")