import os
import torch
from src.datasets.label_file_dataset import LabelFileDataset, CustomConcatDataset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import json


class ImageSingleFolderLoader:
    def __init__(self,
                 preprocess,
                 datasets_folder_name=None,
                 classes_list_path=None,
                 classes_list_to_ignore_path=None,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=14,
                 weighted_sampler=False,
                 class_scanned_percentage_path=None,
                 single_trained_class_label=None
                 ):

        self.datasets_list = []
        for sub_folder in os.listdir(os.path.join(location, datasets_folder_name)):
            sub_folder_path = os.path.join(location, datasets_folder_name, sub_folder)
            if os.path.isdir(sub_folder_path):
                single_dataset = LabelFileDataset(
                    root=sub_folder_path,
                    classes_list_path=classes_list_path,
                    classes_list_ignore_path=classes_list_to_ignore_path,
                    transform=preprocess
                )
                self.datasets_list.append(single_dataset)
        self.dataset = CustomConcatDataset(self.datasets_list)
        
        if weighted_sampler:
            labels = []
            for i in range(len(self.dataset)):
                labels.append(self.dataset.get_label(i))

            # Class_scanned_percentage file is a json file with class index as keys and their weights as values.
            # If a class is not specified in the json file, the default weight for it is 0.05
            if class_scanned_percentage_path is not None:
                with open(class_scanned_percentage_path) as f:
                    class_scanned_percentage = json.load(f)
                sample_weights = [class_scanned_percentage.get(str(label), 0.05) for label in labels]
                print(f"Unique sample_weights for weighte sampler: {set(sample_weights)}")
                num_samples_each_epoch = int(0.66 * len(labels))

            # Balance the train dataset for single trained class. Half of images from this single class,
            # other half of images are from other classes.
            elif single_trained_class_label is not None:
                with open(classes_list_path, "r") as f:
                    total_class_num = len(f.readlines())
                sample_weights = [1.0 if label == single_trained_class_label
                                  else 1/(total_class_num-1) for label in labels]
                num_samples_each_epoch = 2 * sample_weights.count(1.0)

            # Balance the dataset by using the inverse of the percentage of each class as the weight
            else:
                class_counts = Counter(labels)
                print(f"Class counts are {class_counts}")
                min_num_one_class = min(class_counts.values())
                num_classes = len(class_counts)
                # In each epoch, pick up num_classes * min_num_one_class
                num_samples_each_epoch = int(num_classes * min_num_one_class)
                # Calculate weights for each class
                # Inverse of frequency, so more frequent classes get lower weights
                class_weights = {cls: 1 / count for cls, count in class_counts.items()}
                sample_weights = [class_weights[label] for label in labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples_each_epoch, replacement=False)
            print(f"Use WeightedRandomSampler, pick {num_samples_each_epoch} images in each epoch, replacement is False")
            self.data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
            )
        else:
            self.data_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
            )
