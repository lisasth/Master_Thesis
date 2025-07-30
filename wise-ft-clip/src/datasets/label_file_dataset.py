import os
from torchvision.datasets import ImageFolder
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.utils.data import ConcatDataset
import bisect

class LabelFileDataset(ImageFolder):

    def __init__(
        self,
        root: str,
        classes_list_path: str,
        classes_list_ignore_path: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.class_in_all_product_to_idx = {}
        #print(f"class: {classes_list_path}")
        with open(classes_list_path, "r") as f:
            for index, line in enumerate(f):
                self.class_in_all_product_to_idx[line.rstrip()] = index
        self.all_classes = list(self.class_in_all_product_to_idx.keys())
        self.classes_list_ignore_path = classes_list_ignore_path
        super().__init__(
            root = root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory) -> Tuple[List[str], Dict[str, int]]:
        """
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        classes_in_the_folder = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        #print(f"classes in the folder: {classes_in_the_folder}")
        classes_in_all_product = list(self.class_in_all_product_to_idx.keys())
        # for class_in_the_folder in classes_in_the_folder:
        #     if class_in_the_folder not in classes_in_all_product:
        #         print(f"Class {class_in_the_folder} from {directory} not in the product list")
        for class_in_all_product in classes_in_all_product:
            if class_in_all_product not in classes_in_the_folder:
                del self.class_in_all_product_to_idx[class_in_all_product]
                # print(f"Class {class_in_all_product} from the product list not in {directory}")
        if self.classes_list_ignore_path is not None:
            with open(self.classes_list_ignore_path, "r") as f:
                for class_to_ignore in f:
                    if class_to_ignore in self.class_in_all_product_to_idx.keys():
                        del self.class_in_all_product_to_idx[class_to_ignore.rstrip()]
                        print(f"delete {class_to_ignore.rstrip()} in class_in_all_product_to_idx")

        return list(self.class_in_all_product_to_idx.keys()), self.class_in_all_product_to_idx

    def get_label(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: target, where target is class_index of the target class.
        """
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target

class CustomConcatDataset(ConcatDataset):

    def __init__(self, datasets) -> None:
        super().__init__(datasets)

    def get_label(self, idx: int):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: target, where target is class_index of the target class.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_label(sample_idx)
