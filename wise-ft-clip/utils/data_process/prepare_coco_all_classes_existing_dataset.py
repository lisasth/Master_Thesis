import json
from datetime import datetime
import os

width = 818
height = 839
base_dir_to_move = "DE-SBS6/CCTV_mirror/"

root_location = "/home/jingjie/wise-ft-clip/data/Eval_MS_PG_big/MS-PG-old-mixed-big"
save_json_name = "coco-label-file-MS-PG-big.json"
image_file_name_ending = ".jpeg"


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
    if subfolder.name == "Packaged Product" or subfolder.name == "Empty":
        continue
    elif subfolder.is_dir():
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
                coco_file["categories"].append({"name": (label + "/Loose"), "id": category_id + 1})
                coco_file["categories"].append({"name": (label + "/Net"), "id": category_id + 2})
                coco_file["categories"].append({"name": (label + "/Bag"), "id": category_id + 3})

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


# idx = 0

# for blob in container.list_blobs(name_starts_with=root_location):
#     # -- ignore blobs with size 0 ---
#     if blob.get("size", 0) > 0:
#         # -- go up to specified depth --
#         blob_depth = blob.get("name", root_location).count("/")
#         if depth == -1 or blob_depth - root_depth <= depth:
#             idx += 1
#             path = blob["name"][len(root_location) + 1:]
#             if not path.endswith(image_file_name_ending):
#                 continue
#             ean_number = path.split("/")[-2]

#             image_id = len(coco_file["images"]) + 1
#             image = {
#                 "id": image_id,
#                 "width": width,
#                 "height": height,
#                 "file_name": os.path.join(base_dir_to_move, path),
#                 "coco_url": None,
#                 "absolute_url": None,
#                 "date_captured": datetime.now().isoformat(timespec='milliseconds') + "Z"
#             }
#             coco_file["images"].append(image)

#             label = mapping.get(ean_number, default_class_name)
#             annotation_id = len(coco_file["annotations"]) + 1
#             category_id = None
#             for cat in coco_file["categories"]:
#                 if cat["name"] == label:
#                     category_id = cat["id"]
#                     break
#             if category_id is None:
#                 category_id = len(coco_file["categories"]) + 1
#                 coco_file["categories"].append({"name": label, "id": category_id})
#                 coco_file["categories"].append({"name": (label + "/Loose"), "id": category_id + 1})
#                 coco_file["categories"].append({"name": (label + "/Net"), "id": category_id + 2})
#                 coco_file["categories"].append({"name": (label + "/Bag"), "id": category_id + 3})

#             anno = {
#                 "id": annotation_id,
#                 "category_id": category_id,
#                 "image_id": image_id,
#                 "area": 0.0
#             }
#             coco_file["annotations"].append(anno)


# first_categories = ["Packaged Product", "Empty", "Not sure"]
# old_id_name_mapping = {}
# for cat in coco_file["categories"]:
#     old_id_name_mapping[cat["id"]] = cat["name"]

# categories = coco_file["categories"]
# categories_list = sorted([cat["name"] for cat in categories])
# for i, cat in enumerate(first_categories):
#     categories_list.insert(i, categories_list.pop(categories_list.index(cat)))

# new_name_id_mapping = {}
# for i, cat in enumerate(categories_list):
#     new_name_id_mapping[cat] = i + 1

# for annotation in coco_file["annotations"]:
#     annotation["category_id"] = new_name_id_mapping[old_id_name_mapping[annotation["category_id"]]]

# coco_file["categories"] = [{"id": i+1, "name": cat} for i, cat in enumerate(categories_list)]

# with open(save_json_name, 'w') as f:
#     json.dump(coco_file, f, indent=4)

# print(len(coco_file["images"]))
