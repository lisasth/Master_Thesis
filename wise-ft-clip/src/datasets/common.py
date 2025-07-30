import os
import torch
import glob
import collections
import random
from tqdm import tqdm
import torchvision.datasets as torchvision_datasets
import src.datasets as custom_datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
from src.datasets.image_folder_single_dataset_loader import ImageSingleFolderLoader
from src.datasets.sample_file_loader import SampleFileLoader
from torchvision.transforms import transforms
from PIL import Image

from src.models.modeling import TwoViewTransform


class TwoHead(nn.Module):
    def __init__(self, linear_layer, contrastive_output_size=128):
        super().__init__()
        self.linear_layer = linear_layer
        input_size = linear_layer.weight.shape[1]
        self.contrastive_head = nn.Linear(input_size, contrastive_output_size)

    def forward(self, inputs):
        outputs = self.linear_layer(inputs)
        outputs_contrastive = self.contrastive_head(inputs)
        # Normalize the embeddings of contrastive head before output
        outputs_contrastive = F.normalize(outputs_contrastive, dim=1)
        return outputs_contrastive, outputs


def add_contrastive_head(model, contrastive_output_size=128):
    model.fc = TwoHead(model.fc, contrastive_output_size=contrastive_output_size)


def remove_contrastive_head(model):
    model.fc = model.fc.linear_layer


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


class ImageFolderWithPaths(torchvision_datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes - 1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].cuda())

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f'{image_encoder.cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader


def _convert_to_rgb(image):
    return image.convert('RGB')


def build_transform(n_px: int, is_train: bool):
    custom_transforms = [
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.25),
        transforms.RandomAutocontrast(p=0.25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=[0.6, 1.1]),
    ]

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return transforms.Compose([
            *custom_transforms,
            transforms.RandomResizedCrop(n_px, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])

def get_dataset(args, is_train):

    if is_train:
        dataset_class = getattr(custom_datasets, args.train_dataset)
        datasets_folder_name = args.train_set
        weighted_sampler = args.weighted_sampler
        if dataset_class is ImageSingleFolderLoader and args.single_trained_class_index is not None:
            single_trained_class_label = args.single_trained_class_index - 1
        else:
            single_trained_class_label = None
        class_scanned_percentage_path = args.class_scanned_percentage_path
        sample_file = args.train_sample_file_path
        preprocess_fn = build_transform(n_px=224, is_train=True)
        if args.contrastive:
            transform_train = transforms.Compose([*preprocess_fn.transforms])
            preprocess_fn.transforms = [TwoViewTransform(transform_train)]

    else:
        dataset_class = getattr(custom_datasets, args.eval_datasets)
        datasets_folder_name = args.eval_set
        weighted_sampler = False
        single_trained_class_label = None
        class_scanned_percentage_path = None
        sample_file = args.eval_sample_file_path
        preprocess_fn = build_transform(n_px=224, is_train=False) 


    if dataset_class is ImageSingleFolderLoader:
        dataset = dataset_class(
            datasets_folder_name=datasets_folder_name,
            preprocess=preprocess_fn,
            classes_list_path=args.classes_list_path,
            location=args.data_location,
            batch_size=args.batch_size,
            weighted_sampler=weighted_sampler,
            class_scanned_percentage_path=class_scanned_percentage_path,
            single_trained_class_label=single_trained_class_label,
        )
    elif dataset_class is SampleFileLoader:
        dataset = dataset_class(
            preprocess=preprocess_fn,
            classes_list_path=args.classes_list_path,
            sample_file=sample_file,
            batch_size=args.batch_size
        )
    else:
        raise ValueError("Please use implemented dataloader")

    return dataset
