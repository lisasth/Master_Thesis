import os
import random
import shutil


def random_select(root, class_name, num_to_be_selected, save_path):

    class_path = os.path.join(root, class_name)
    all_images = os.listdir(class_path)
    
    if len(all_images) < num_to_be_selected:
        print(f'Class {class_name} does not have enough images to select {num_to_be_selected} images')
        num_to_be_selected = len(all_images)

    selected_images = random.sample(all_images, num_to_be_selected)
    
    # Create a destination directory for the selected images
    destination_folder = os.path.join(save_path, class_name)
    os.makedirs(destination_folder, exist_ok=True)
    
    # Copy the selected images to the destination folder
    for image in selected_images:
        source_path = os.path.join(class_path, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.copy(source_path, destination_path)

def count_file(root):
    subfolder_to_num_files = {}
    for subfolder in os.scandir(root):
        if subfolder.is_dir():
            files = next(os.walk(subfolder.path))[2]
            num_files = len(files)
            subfolder_to_num_files[subfolder.name] = num_files
    return subfolder_to_num_files

if __name__ == "__main__":
    folder_to_be_selected = "/home/jingjie/wise-ft-clip/data/UK-1913/old-mixed"
    folder_to_save = "/home/jingjie/wise-ft-clip/data/UK-1913/old-mixed-select-200"
    num_to_be_selected = 200

    class_to_num_files_source_folder = count_file(folder_to_be_selected)
    # class_to_num_files_smart_sco = count_file("data/smart_sco_frames")
    # nums_in_smart_sco = list(class_to_num_files_smart_sco.values())
    # avg_nums_in_smart_sco = int(sum(nums_in_smart_sco)/len(nums_in_smart_sco))
    for class_name, _ in class_to_num_files_source_folder.items():
        # num_in_smart_sco = class_to_num_files_smart_sco.get(class_name)
        # if num_in_smart_sco is None:
        #     num_to_be_selected = avg_nums_in_smart_sco
        # else:
        #     num_to_be_selected = num_in_smart_sco
        
        # !!!Give fixed num to be selected!!!
        num_to_be_selected = num_to_be_selected


        random_select(root=folder_to_be_selected, 
                      class_name=class_name, 
                      num_to_be_selected=num_to_be_selected, 
                      save_path=folder_to_save
                      )


# 先count，输出dict，然后根据dict做筛选的算法，然后check是不是top down少于其他的相加，最后random select