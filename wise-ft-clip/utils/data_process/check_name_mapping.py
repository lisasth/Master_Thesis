import os
import csv

if __name__ == "__main__":

    all_product_list_path = "/home/jingjie/wise-ft-clip/class_names_list/classes_2024_07_15_uk_hersham.txt"
    # csv_to_check_path = "csv_files/class_to_renamed_class.csv"
    folder_to_check_path = "/home/jingjie/wise-ft-clip/data/Train_UK_cleaned"
    all_product_list = []
    with open(all_product_list_path, "r") as f:
        for line in f:
            all_product_list.append(line.rstrip())
    print(all_product_list)

    all_class_names = []
    for subfolder_name in os.listdir(folder_to_check_path):
        if os.path.isdir(os.path.join(folder_to_check_path, subfolder_name)):
            for class_name in os.listdir(os.path.join(folder_to_check_path, subfolder_name)):
                if os.path.isdir(os.path.join(folder_to_check_path, subfolder_name,class_name)):
                    if class_name not in all_class_names:
                        all_class_names.append(class_name)
    all_class_names = sorted(all_class_names)
    print(all_class_names)


    for class_name in all_class_names:
        if class_name not in all_product_list:
            print(f"class_name {class_name} not in all_product_list")
    

    print("---------------------------------------")

    for product_name in all_product_list:
        if product_name not in all_class_names:
            print(f"product_name {product_name} not in the folder")