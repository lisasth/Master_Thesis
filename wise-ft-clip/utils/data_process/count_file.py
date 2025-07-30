import os
import argparse
import csv

def main(args):
    csv_folder_name = args.csv_folder_name
    class_list = []
    with open(args.path_to_class_list, "r") as f:
        for line in f:
            class_list.append(line.rstrip())
    subfolder_to_total_num_files = {}
    for folder_to_count in os.listdir(args.folder):
        if os.path.isdir(os.path.join(args.folder, folder_to_count)):
            subfolder_to_num_files = {}
            for subfolder in os.scandir(os.path.join(args.folder, folder_to_count)):
                if subfolder.is_dir() and subfolder.name in class_list:
                    files = next(os.walk(subfolder.path))[2]
                    num_files = len(files)
                    subfolder_to_num_files[subfolder.name] = num_files
            subfolder_to_num_files = dict(sorted(subfolder_to_num_files.items()))
            print("----------------------------------------------")
            print(folder_to_count)
            print(subfolder_to_num_files)
            print("Num of subfolders", len(subfolder_to_num_files))
            csv_file_name = folder_to_count.split("/")[-1] + "_class_numbers.csv"
            csv_path = os.path.join(csv_folder_name, csv_file_name)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            with open(csv_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                for subfolder, num_files in subfolder_to_num_files.items():
                    writer.writerow([subfolder, str(num_files)])
            
            for k in subfolder_to_num_files.keys():
                if subfolder_to_total_num_files.get(k) is None:
                    subfolder_to_total_num_files[k] = subfolder_to_num_files[k]
                else:
                    subfolder_to_total_num_files[k] += subfolder_to_num_files[k]
    print("----------------------All folders------------------------")
    subfolder_to_total_num_files = dict(sorted(subfolder_to_total_num_files.items()))
    print(subfolder_to_total_num_files)
    print("Num of subfolders in all folders", len(subfolder_to_total_num_files))

    csv_file_name = "all_class_numbers.csv"
    csv_path = os.path.join(csv_folder_name, csv_file_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for subfolder, num_files in subfolder_to_total_num_files.items():
            writer.writerow([subfolder, str(num_files)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--path_to_class_list")
    parser.add_argument("--csv_folder_name")
    args = parser.parse_args()
    main(args)
