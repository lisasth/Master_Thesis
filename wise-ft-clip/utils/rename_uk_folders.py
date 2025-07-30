import os
import csv

uk_data_root = "/home/jingjie/wise-ft-clip/data/Eval_UK_cleaned"
uk_to_german_name_mapping_path = "/home/jingjie/wise-ft-clip/data/uk_to_german_fnv_name_mapping.csv"
uk_to_german_name_mapping = {}
with open(uk_to_german_name_mapping_path, "r") as f:
    reader = csv.reader(f, delimiter=";")
    for line in reader:
        uk_to_german_name_mapping[line[0]] = line[1]

for dataset_folder in os.listdir(uk_data_root):
    if os.path.isdir(os.path.join(uk_data_root, dataset_folder)):
        for fnv_class in os.listdir(os.path.join(uk_data_root,dataset_folder)):
            if os.path.isdir(os.path.join(uk_data_root,dataset_folder,fnv_class)):
                try:
                    mapped_german_fnv_name = uk_to_german_name_mapping[fnv_class]
                    os.rename(os.path.join(uk_data_root,dataset_folder,fnv_class), os.path.join(uk_data_root,dataset_folder,mapped_german_fnv_name))
                except:
                    print(f"{fnv_class} not in the list, \n"
                          f"{os.path.join(uk_data_root,dataset_folder,fnv_class)} not possible to be renamed")