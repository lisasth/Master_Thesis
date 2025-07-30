import os
import shutil

source_folder = '/home/jingjie/wise-ft-clip/data/Train_UK_cleaned/UK-cleaned-2024_03_26-loose-sco2/Medium Avocado_all'
destination_folder = '/home/jingjie/wise-ft-clip/data/Train_UK_cleaned/UK-cleaned-2024_03_26-loose-sco2/Aubergine'


os.makedirs(destination_folder, exist_ok=True)


def is_image_file(filename):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    return filename.lower().endswith(image_extensions)


def get_every_second_image(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if is_image_file(f)])
    for i, image_file in enumerate(image_files):
        if i % 13 == 0:
            yield os.path.join(folder_path, image_file)

for file_name in get_every_second_image(source_folder):
    file_path = os.path.join(source_folder, file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, destination_folder)
