import os
import csv

if __name__ == "__main__":

    all_product_list_path = "/home/jingjie/wise-ft-clip/class_names_list/classes_in_old_dataset_renamed_add_cctv_mirror_20240212renamed.txt"
    # csv_to_check_path = "csv_files/class_to_renamed_class.csv"
    folder_to_check_path = "/home/jingjie/wise-ft-clip/data/UK-1913/old-mixed"
    all_product_list = []
    with open(all_product_list_path, "r") as f:
        for line in f:
            all_product_list.append(line.rstrip())
    print(all_product_list)

    all_subfolder_names = []
    for subfolder_name in os.listdir(folder_to_check_path):
        if os.path.isdir(os.path.join(folder_to_check_path, subfolder_name)):
            all_subfolder_names.append(subfolder_name)
    all_subfolder_names = sorted(all_subfolder_names)


    for subfolder_name in all_subfolder_names:
        if subfolder_name not in all_product_list:
            print(f"subfolder_name {subfolder_name} not in all_product_list")
    

    print("---------------------------------------")

    for product_name in all_product_list:
        if product_name not in all_subfolder_names:
            print(f"product_name {product_name} not in the folder")