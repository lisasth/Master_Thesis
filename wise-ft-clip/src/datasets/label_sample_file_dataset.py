from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.folder import has_file_allowed_extension, default_loader
import json
from torch.utils.data import Dataset


class LabelSampleFileDataset(Dataset):

    def __init__(
            self,
            classes_list_path: str,
            sample_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.class_in_all_product_to_idx = {}
        with open(classes_list_path, "r") as f:
            for index, line in enumerate(f):
                self.class_in_all_product_to_idx[line.rstrip()] = index
        self.all_classes = list(self.class_in_all_product_to_idx.keys())
        self.sample_file = sample_file
        classes, class_to_idx = self.find_classes()
        samples = self.make_dataset(class_to_idx, self.sample_file)
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        with open(self.sample_file, 'r') as f:
            sample_data = json.load(f)
        classes_in_all_product = list(self.class_in_all_product_to_idx.keys())
        classes_in_sample_file = list(sample_data.keys())
        for class_in_all_product in classes_in_all_product:
            if class_in_all_product not in classes_in_sample_file:
                del self.class_in_all_product_to_idx[class_in_all_product]

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

    @staticmethod
    def make_dataset(
            class_to_idx: Dict[str, int],
            sample_file: str
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        Args:
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            sample_file (str): Path to the sample file containing classes and paths for images in each class

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        instances = []
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
            for class_name in sample_data.keys():
                class_index = class_to_idx.get(class_name)
                samples_paths = sample_data[class_name]
                if class_index is None:
                    raise ValueError(f"{class_name} is contained in sample file that is not in the class list")
                for path in samples_paths:
                    # Check if the path is a valid image file
                    valid_iamge_extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
                    if has_file_allowed_extension(path, valid_iamge_extensions):
                        instances.append((path, class_index))

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
