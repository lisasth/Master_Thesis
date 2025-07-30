import json
from datetime import datetime
import os

width = 450
height = 300
base_dir_to_move = "DE-SBS6/CCTV_mirror/2024_07_24-bag/"

root_location = "/Users/guoji/Movies/20240724_pre_labelled/2024_07_24-bag"
save_json_name = "coco-label-file-2024_07_24-bag.json"
image_file_name_ending = ".png"


coco_file = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "Not sure"
        }
    ]
}

for subfolder in os.scandir(root_location):
    if subfolder.is_dir():
        for image in os.scandir(subfolder.path):
            if not image.name.endswith(image_file_name_ending):
                continue
            
            image_id = len(coco_file["images"]) + 1
            image = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": os.path.join(base_dir_to_move, subfolder.name, image.name),
                "coco_url": None,
                "absolute_url": None,
                "date_captured": datetime.now().isoformat(timespec='milliseconds') + "Z"
            }
            coco_file["images"].append(image)

            label = subfolder.name
            annotation_id = len(coco_file["annotations"]) + 1
            category_id = None
            for cat in coco_file["categories"]:
                if cat["name"] == label:
                    category_id = cat["id"]
                    break
            if category_id is None:
                category_id = len(coco_file["categories"]) + 1
                coco_file["categories"].append({"name": label, "id": category_id})

            anno = {
                "id": annotation_id,
                "category_id": category_id,
                "image_id": image_id,
                "area": 0.0
            }
            coco_file["annotations"].append(anno)

first_categories = ["Not sure"]
old_id_name_mapping = {}
for cat in coco_file["categories"]:
    old_id_name_mapping[cat["id"]] = cat["name"]

categories = coco_file["categories"]
categories_list = sorted([cat["name"] for cat in categories])
for i, cat in enumerate(first_categories):
    categories_list.insert(i, categories_list.pop(categories_list.index(cat)))

new_name_id_mapping = {}
for i, cat in enumerate(categories_list):
    new_name_id_mapping[cat] = i + 1

for annotation in coco_file["annotations"]:
    annotation["category_id"] = new_name_id_mapping[old_id_name_mapping[annotation["category_id"]]]

coco_file["categories"] = [{"id": i+1, "name": cat} for i, cat in enumerate(categories_list)]

with open(save_json_name, 'w') as f:
    json.dump(coco_file, f, indent=4)

print(len(coco_file["images"]))