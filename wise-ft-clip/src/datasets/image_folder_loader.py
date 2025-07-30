import os
import torch
from src.datasets.label_file_dataset import LabelFileDataset, CustomConcatDataset
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from collections import Counter
import json


class ImageFolderLoader:
    def __init__(self,
                 train_set,
                 eval_set,
                 preprocess,
                 classes_list_path,
                 classes_list_ignore_during_val_path=None,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=14,
                 weighted_sampler=False,
                 class_scanned_percentage_path=None,
                 single_trained_class_label=None):

        train_set = train_set
        self.train_datasets = []
        for train_folder in os.listdir(os.path.join(location, train_set)):
            train_folder_path = os.path.join(location, train_set, train_folder)
            if os.path.isdir(train_folder_path):
                train_dataset = LabelFileDataset(
                    root=train_folder_path,
                    classes_list_path=classes_list_path,
                    transform=preprocess
                )
                self.train_datasets.append(train_dataset)
        self.train_dataset = CustomConcatDataset(self.train_datasets)

        if weighted_sampler:
            labels = []
            for i in range(len(self.train_dataset)):
                labels.append(self.train_dataset.get_label(i))

            if class_scanned_percentage_path is not None:
                with open(class_scanned_percentage_path) as f:
                    class_scanned_percentage = json.load(f)
                sample_weights = [class_scanned_percentage.get(str(label), 0.05) for label in labels]
                print(f"Unique sample_weights for weighte sampler: {set(sample_weights)}")
                num_samples_each_epoch = int(0.66 * len(labels))
            # Balance the train dataset for single trained class. Half of images from this single class, other half of images are from othere classes.
            elif single_trained_class_label is not None:
                with open(classes_list_path, "r") as f:
                    total_class_num = len(f.readlines())
                sample_weights = [1.0 if label == single_trained_class_label else 1 / (total_class_num - 1) for label in
                                  labels]
                num_samples_each_epoch = 2 * sample_weights.count(1.0)
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
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples_each_epoch,
                                            replacement=False)
            print(
                f"Use WeightedRandomSampler, pick {num_samples_each_epoch} images in each epoch, replacement is False")
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
            )

        eval_set = eval_set
        self.test_datasets = []
        for test_folder in os.listdir(os.path.join(location, eval_set)):
            test_folder_path = os.path.join(location, eval_set, test_folder)
            if os.path.isdir(test_folder_path):
                test_dataset = LabelFileDataset(
                    root=test_folder_path,
                    classes_list_path=classes_list_path,
                    classes_list_ignore_path=classes_list_ignore_during_val_path,
                    transform=preprocess
                )
            self.test_datasets.append(test_dataset)
        self.test_dataset = torch.utils.data.ConcatDataset(self.test_datasets)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        )

        # self.test_dataset.all_classes includes all classes in the product list,
        # even if some of them may not appear in the folder of train/eval set
        # self.classnames = self.test_dataset.all_classes
