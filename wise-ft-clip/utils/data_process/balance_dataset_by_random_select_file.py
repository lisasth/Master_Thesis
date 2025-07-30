import os
import random
import shutil
import copy


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


def get_priority(num_dict):
    num_dict_with_priority = {}
    for dataset_name, num in num_dict.items():
        dataset_name_lower = dataset_name.lower()
        priority = None
        if ("2024_07" in dataset_name_lower 
            or "2024_08" in dataset_name_lower
            or "2024_09" in dataset_name_lower
            or "2024_10" in dataset_name_lower) and "sco" in dataset_name_lower:
            priority = 1
        elif "top_down_uk" in dataset_name_lower:
            priority = 2
        elif "uk-cleaned" in dataset_name_lower:
            priority = 3
        elif "top_down" in dataset_name_lower and "uk" not in dataset_name_lower:
            priority = 5
        else:
            priority = 4
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

    for priority in sorted_priorities:
        priority_datasets = priority_map[priority]
        images_to_select = int(remaining_images_needed * percentage)
        
        if priority == sorted_priorities[-1]:  # If it's the last priority, take all remaining
            images_to_select = remaining_images_needed
        
        total_images_in_priority = sum(datasets_dict[name]["number of images"] for name, _ in priority_datasets)
        selected_from_this_priority = 0
        
        for dataset_name, num_images in priority_datasets:
            weight = datasets_dict[dataset_name]["number of images"] / total_images_in_priority
            images_for_dataset = int(images_to_select * weight)

            # Adjust if the calculated number exceeds available images
            if images_for_dataset > datasets_dict[dataset_name]["number of images"]:
                images_for_dataset = datasets_dict[dataset_name]["number of images"]
            datasets_dict[dataset_name]["images_to_select"] = images_for_dataset
            selected_from_this_priority += images_for_dataset
        
        remaining_images_needed -= selected_from_this_priority
        if remaining_images_needed <= 0:
            break
    return datasets_dict


def check_top_down_less_than_cctv(datasets_dict, percentage_each_priority):
    top_down_select_num = 0
    top_down_total_num = 0
    cctv_mirror_select_num = 0
    cctv_mirror_total_num = 0
    for dataset_name, info in datasets_dict.items():
        if "top_down" in dataset_name:
            top_down_select_num += info["images_to_select"]
            top_down_total_num += info["number of images"]
        else:
            cctv_mirror_select_num += info["images_to_select"]
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


if __name__ == "__main__":
    root = "/home/jingjie/wise-ft-clip/data/Train_UK_cleaned"
    total_images_needed_original = 349
    dst_root = f"/home/jingjie/wise-ft-clip/data/Train_UK_cleaned_select_{total_images_needed_original}_each_pack"
    path_to_class_list = "/home/jingjie/wise-ft-clip/class_names_list/classes_2024_08_26_uk_hersham.txt"
    valid_image_extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

    class_list = []
    with open(path_to_class_list, "r") as f:
        for line in f:
            class_list.append(line.rstrip())

    for class_name in class_list:
        percentage_each_priority = 0.8
        dataset_name_to_num_files = count_file(root=root, class_name=class_name)
        pack_type_to_dataset_name_to_num_files = get_pack_type_to_dataset_name_to_num_files(dataset_name_to_num_files)
        for pack_name, pack_dict in pack_type_to_dataset_name_to_num_files.items():
            total_images_needed = total_images_needed_original
            if pack_name == "no_pack_type":
                total_images_needed = total_images_needed * 3
            if class_name == "Empty" and pack_name != "no_pack_type":
                continue
            if class_name in ["Almond", "Cashew", "Pistachio"]:
                total_images_needed = total_images_needed * 3 / 2
            print("------------------------------------")
            print(class_name, pack_name)
            pack_dict = get_priority(pack_dict)
            pack_dict = calculate_images_to_select(pack_dict, total_images_needed, percentage_each_priority)
            check_top_down_less_than_cctv(pack_dict, percentage_each_priority)

            for dataset_name, info in pack_dict.items():
                src_path = os.path.join(root, dataset_name, class_name)
                dst_path = os.path.join(dst_root, dataset_name, class_name)
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                os.makedirs(dst_path)
                if "images_to_select" in info and info["images_to_select"] > 0:
                    image_files = os.listdir(src_path)
                    selected_files = random.sample(image_files, info["images_to_select"])
                    len_selected_files = len(selected_files)

                    for image_file in selected_files:
                        file_name_extension = os.path.splitext(image_file)[1]
                        if file_name_extension not in valid_image_extensions:
                            print(f"{os.path.join(src_path, image_file)} is not an image")
                            len_selected_files -= 1
                            if len_selected_files == 0:
                                shutil.rmtree(dst_path)
                            continue
                        src = os.path.join(src_path, image_file)
                        dst = os.path.join(dst_path, image_file)
                        shutil.copy(src, dst)