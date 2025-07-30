import torch
from src.datasets.label_sample_file_dataset import LabelSampleFileDataset
from torch.utils.data import ConcatDataset


class SampleFileLoader:
    def __init__(self,
                 preprocess,
                 classes_list_path=None,
                 sample_file: str = None,
                 batch_size=128,
                 num_workers=8
                 ):
        self.dataset = LabelSampleFileDataset(
            classes_list_path=classes_list_path,
            sample_file=sample_file,
            transform=preprocess
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        )