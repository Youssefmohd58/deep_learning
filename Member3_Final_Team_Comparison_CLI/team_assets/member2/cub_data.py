"""Reusable CUB-200-2011 classification dataset and loaders (Member 1, Milestone 1).

Usage:
    from src.data.cub_data import make_classification_loaders
    train_loader, val_loader, test_loader = make_classification_loaders(
        train, test, manifest_path='outputs/member1_eda/train_validation_split.csv'
    )

No downloading is done at module import time. Images are decoded on demand.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 128                 # 128 is ~3x faster than 224 and fine for a scratch CNN
RESIZE_SHORTER_SIDE_RATIO = 256 / 224   # eval: resize to size*ratio, then center-crop
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size=IMAGE_SIZE, train=True, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                   crop_scale=(0.35, 1.00)):
    """Return training-only augmentation or deterministic evaluation preprocessing."""
    if train:
        steps = [
            T.Lambda(lambda im: im.convert('RGB')),
            T.RandomResizedCrop(image_size, scale=crop_scale, ratio=(0.75, 1.33),
                                interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.TrivialAugmentWide(),
        ]
        tail = [T.ToTensor(), T.Normalize(mean=mean, std=std), T.RandomErasing(p=0.25)]
    else:
        steps = [
            T.Lambda(lambda im: im.convert('RGB')),
            T.Resize(round(image_size * RESIZE_SHORTER_SIDE_RATIO),
                     interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
        ]
        tail = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(steps + tail)


class CUBClassificationDataset(Dataset):
    """Subset of a Hugging Face dataset referenced by ORIGINAL row indices."""

    def __init__(self, hf_dataset, indices=None, transform=None, label_to_index=None):
        # Only image + label are needed; dropping the long caption column makes row access faster.
        if hasattr(hf_dataset, 'select_columns'):
            keep = [c for c in ('image', 'label') if c in hf_dataset.column_names]
            hf_dataset = hf_dataset.select_columns(keep)
        self.data = hf_dataset
        self.indices = list(range(len(hf_dataset))) if indices is None else [int(i) for i in indices]
        self.transform = transform or get_transforms(train=False)
        self.label_to_index = label_to_index
        if not all(0 <= i < len(hf_dataset) for i in self.indices):
            raise ValueError('Row indices must be within the Hugging Face dataset')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        source_index = self.indices[ix]
        item = self.data[source_index]
        raw_label = item['label']
        if self.label_to_index is None:
            label = int(raw_label)
        else:
            label = self.label_to_index[raw_label]
        return {
            'image': self.transform(item['image']),
            'label': torch.tensor(label, dtype=torch.long),
            'row_index': torch.tensor(source_index, dtype=torch.long),
        }


def _seed_worker(worker_id):
    # Reproducible NumPy/Python randomness when num_workers > 0.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _validate_manifest(manifest, train_hf):
    required = {'row_index', 'split', 'label'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Manifest missing columns {required - set(manifest.columns)}')
    if manifest['row_index'].duplicated().any():
        raise ValueError('Manifest has duplicated source image indices')
    if set(manifest['split']) != {'train', 'validation'}:
        raise ValueError('Manifest must contain both train and validation splits only')
    source_indices = manifest['row_index'].astype(int).tolist()
    if any(i < 0 or i >= len(train_hf) for i in source_indices):
        raise ValueError('Manifest contains out-of-bounds indices')


def make_classification_loaders(train_hf, test_hf, manifest_path,
                                batch_size=32, num_workers=0, seed=42,
                                image_size=IMAGE_SIZE,
                                mean=IMAGENET_MEAN, std=IMAGENET_STD,
                                allowed_test_indices=None,
                                crop_scale=(0.35, 1.00)):
    """Build train/validation/test loaders using a saved image-level split manifest.

    Labels must be numeric and consistently encoded between official splits.
    An optional list of valid test indices excludes previously identified decode failures.
    """
    manifest = pd.read_csv(Path(manifest_path))
    _validate_manifest(manifest, train_hf)

    raw_train_labels = train_hf['label']  # metadata only; avoids decoding thousands of images
    source_indices = manifest['row_index'].astype(int).tolist()
    actual_labels = [raw_train_labels[i] for i in source_indices]
    if any(str(a) != str(b) for a, b in zip(actual_labels, manifest['label'])):
        raise ValueError('Manifest labels no longer match dataset: wrong dataset/version?')

    # Shared class-to-index mapping supports either integer labels or species-name strings.
    distinct = sorted(set(raw_train_labels), key=lambda x: (str(type(x)), str(x)))
    if all(isinstance(x, (int, np.integer)) for x in distinct) and set(distinct) == set(range(len(distinct))):
        label_to_index = None  # existing class IDs 0,...,C-1
    else:
        label_to_index = {name: i for i, name in enumerate(distinct)}
    if not set(test_hf['label']).issubset(set(distinct)):
        raise ValueError('Official test labels are not covered by the training class mapping')

    train_ids = manifest.loc[manifest['split'] == 'train', 'row_index'].astype(int).tolist()
    val_ids = manifest.loc[manifest['split'] == 'validation', 'row_index'].astype(int).tolist()
    if not train_ids or not val_ids:
        raise ValueError('Both train and validation must be nonempty')
    if allowed_test_indices is None:
        allowed_test_indices = list(range(len(test_hf)))

    training = CUBClassificationDataset(
        train_hf, train_ids,
        get_transforms(image_size, True, mean, std, crop_scale), label_to_index=label_to_index)
    validation = CUBClassificationDataset(
        train_hf, val_ids,
        get_transforms(image_size, False, mean, std), label_to_index=label_to_index)
    testing = CUBClassificationDataset(
        test_hf, allowed_test_indices,
        get_transforms(image_size, False, mean, std), label_to_index=label_to_index)

    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
                  persistent_workers=num_workers > 0)
    return (
        DataLoader(training, shuffle=True, generator=generator, **common),
        DataLoader(validation, shuffle=False, **common),
        DataLoader(testing, shuffle=False, **common),
    )