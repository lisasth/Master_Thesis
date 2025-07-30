import csv
import os

ROOT_PATH = "/home/jingjie/wise-ft-clip/data/UK-HQ/CCTV_mirror"



for dataset_folder in os.listdir(ROOT_PATH):
    if os.path.isdir(os.path.join(ROOT_PATH, dataset_folder)):
        csv_file_path = os.path.join(ROOT_PATH, dataset_folder+".csv")
        print(csv_file_path)
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dataset_folder])
            for fnv_class in os.listdir(os.path.join(ROOT_PATH,dataset_folder)):
                if os.path.isdir(os.path.join(ROOT_PATH,dataset_folder,fnv_class)):
                    writer.writerow([fnv_class])
