import os
import argparse
import csv

def main(args):
    class_list = []
    with open(args.path_to_class_list, "r") as f:
        for line in f:
            class_list.append(line.rstrip())
    csv_folder_name = args.csv_folder_name
    subfolder_to_total_num_files = {}
    for folder_to_count in os.scandir(args.root_folder):

        folder_to_count = folder_to_count.path
        packing_tpye = "No Packing Type"
        if "loose" in folder_to_count:
            packing_tpye = "Loose"
        elif "bag" in folder_to_count:
            packing_tpye = "Bag"
        elif "net" in folder_to_count:
            packing_tpye = "Net"

        if os.path.isdir(folder_to_count):
            subfolder_to_num_files = {}
            for subfolder in os.scandir(folder_to_count):
                if subfolder.is_dir() and subfolder.name in class_list:
                    files = next(os.walk(subfolder.path))[2]
                    num_files = len(files)
                    subfolder_to_num_files[subfolder.name] = num_files
                else:
                    print(f"{subfolder.name} not in {args.path_to_class_list}")
            subfolder_to_num_files = dict(sorted(subfolder_to_num_files.items()))

            if len(subfolder_to_num_files) != 0:
                print("----------------------------------------------")
                print(folder_to_count)
                print(subfolder_to_num_files)
                csv_file_name = folder_to_count.split("/")[-1] + "_class_numbers.csv"
                csv_path = os.path.join("csv_files_with_packing_types", csv_folder_name, csv_file_name)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)

                with open(csv_path, "w") as csv_file:
                    writer = csv.writer(csv_file, delimiter=';')
                    for subfolder, num_files in subfolder_to_num_files.items():
                        writer.writerow([subfolder, str(num_files)])

                for k in subfolder_to_num_files.keys():
                    if subfolder_to_total_num_files.get(k) is None:
                        subfolder_to_total_num_files[k] = {}
                        subfolder_to_total_num_files[k][packing_tpye] = subfolder_to_num_files[k]
                    else:
                        if subfolder_to_total_num_files[k].get(packing_tpye) is None:
                            subfolder_to_total_num_files[k][packing_tpye] = subfolder_to_num_files[k]
                        else:
                            subfolder_to_total_num_files[k][packing_tpye] += subfolder_to_num_files[k]
    print("----------------------All folders------------------------")
    subfolder_to_total_num_files = dict(sorted(subfolder_to_total_num_files.items()))
    print(subfolder_to_total_num_files)
    print("Num of subfolders in all folders", len(subfolder_to_total_num_files))

    csv_file_name = f"all_class_numbers_with_packing_types_{csv_folder_name}.csv"
    csv_path = os.path.join("csv_files_with_packing_types", csv_folder_name, csv_file_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["Class Name", "Loose", "Bag", "Net", "No Packing Type"])
        for subfolder, num_files in subfolder_to_total_num_files.items():
            writer.writerow([subfolder, 
                             num_files.get("Loose", 0),
                             num_files.get("Bag", 0),
                             num_files.get("Net", 0),
                             num_files.get("No Packing Type", 0),
                            ])
    

    missing_class_file_save_path = os.path.join("csv_files_with_packing_types", csv_folder_name, f"missing_classes_in_{os.path.basename(args.root_folder)}.txt")
    with open(missing_class_file_save_path, "w") as f:
        for class_name in class_list:
            if class_name not in subfolder_to_total_num_files.keys():
                print(f"{class_name} not in {args.root_folder}")
                f.write(f"{class_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder")
    parser.add_argument("--csv_folder_name")
    parser.add_argument("--path_to_class_list")
    args = parser.parse_args()
    main(args)
