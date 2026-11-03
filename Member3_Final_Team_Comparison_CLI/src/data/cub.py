"""CUB-200 preprocessing, data loading and team-aligned train/validation split.

The final team project uses a saved manifest supplied with Member 2's checkpoint.
A separately generated split must NOT silently overwrite that authoritative file.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def transforms_for(train, image_size=IMAGE_SIZE):
    """Training augmentations versus deterministic validation/inference transforms."""
    if train:
        steps = [T.Lambda(lambda im: im.convert('RGB')),
                 T.RandomResizedCrop(image_size, scale=(0.80, 1.00), ratio=(0.85, 1.15),
                                    interpolation=InterpolationMode.BILINEAR),
                 T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10),
                 T.ColorJitter(brightness=0.10, contrast=0.10)]
    else:
        steps = [T.Lambda(lambda im: im.convert('RGB')),
                 T.Resize(round(image_size * 256 / 224), interpolation=InterpolationMode.BILINEAR),
                 T.CenterCrop(image_size)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class CUBClassificationDataset(Dataset):
    """Lazily decode only the images requested by the PyTorch DataLoader."""
    def __init__(self, dataset, indices, transform):
        self.data = dataset
        self.indices = [int(i) for i in indices]
        self.transform = transform
        if any(i < 0 or i >= len(dataset) for i in self.indices):
            raise ValueError('CUB original row index out of range')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        row_index = self.indices[ix]
        item = self.data[row_index]
        return {
            'image': self.transform(item['image']),
            'label': torch.tensor(int(item['label']), dtype=torch.long),
            'row_index': torch.tensor(row_index, dtype=torch.long),
        }


def get_class_names(train_ds, test_ds=None):
    from datasets import ClassLabel
    if isinstance(train_ds.features['label'], ClassLabel):
        names = list(train_ds.features['label'].names)
        if test_ds is not None and isinstance(test_ds.features['label'], ClassLabel):
            if names != list(test_ds.features['label'].names):
                raise ValueError('Train/test label names do not match')
        return names
    names = {}
    for label, filename in zip(train_ds['label'], train_ds['file_name']):
        species = str(filename).rsplit('_', 2)[0].replace('_', ' ')
        if int(label) in names and names[int(label)] != species:
            raise ValueError(f'Inconsistent species for class {label}')
        names[int(label)] = species
    return [names[i] for i in range(200)]


def get_or_create_split(train_ds, cfg):
    """Persist the exact random stratified 80/20 partition used by this project."""
    if cfg.get('split_manifest'):
        path=Path(cfg['split_manifest'])
        if not path.is_file():
            raise FileNotFoundError(f'Team split manifest is missing: {path}')
    else:
        path = Path(cfg['output_dir']) / 'data' / 'train_validation_split.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(train_ds['label'], dtype=int)
    if not path.exists():
        all_rows = np.arange(len(train_ds), dtype=int)
        train_rows, val_rows = train_test_split(
            all_rows, test_size=float(cfg['validation_fraction']),
            random_state=int(cfg['seed']), stratify=labels)
        assignments = np.full(len(all_rows), 'train', dtype=object)
        assignments[val_rows] = 'validation'
        pd.DataFrame({'row_index': all_rows, 'split': assignments,
                      'label': labels}).to_csv(path, index=False)
        print('Created reproducible train/validation split:', path)
    manifest = pd.read_csv(path)
    needed = {'row_index', 'split', 'label'}
    if not needed.issubset(manifest.columns):
        raise ValueError(f'Split manifest is missing {needed-set(manifest.columns)}')
    if len(manifest) != len(labels) or manifest['row_index'].duplicated().any():
        raise ValueError('Split manifest rows are missing or duplicated')
    rows = manifest['row_index'].astype(int).to_numpy()
    if set(rows) != set(range(len(labels))) or set(manifest['split']) != {'train','validation'}:
        raise ValueError('Invalid split labels or original row indices')
    if not np.array_equal(manifest['label'].to_numpy(dtype=int), labels[rows]):
        raise ValueError('Split manifest class labels differ from the public dataset')
    train_ids = manifest.loc[manifest['split']=='train','row_index'].astype(int).tolist()
    val_ids = manifest.loc[manifest['split']=='validation','row_index'].astype(int).tolist()
    if not train_ids or not val_ids:
        raise ValueError('Train and validation sets must be nonempty')
    if set(labels[train_ids]) != set(range(200)) or set(labels[val_ids]) != set(range(200)):
        raise ValueError('Each split must cover all 200 bird classes')
    return train_ids, val_ids, hashlib.sha256(path.read_bytes()).hexdigest()


def make_loader(dataset, batch_size, shuffle, num_workers=0, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator,
                      pin_memory=torch.cuda.is_available())


def stratified_subset(dataset, n, labels, seed=42):
    """Reduce tuning compute without changing the fixed train/validation split."""
    if n >= len(dataset):
        return dataset
    if n < 200:
        raise ValueError('Tuning subset requires >= 200 images')
    original_ids = np.asarray(dataset.indices, dtype=int)
    local_labels = np.asarray(labels, dtype=int)[original_ids]
    subset_positions, _ = train_test_split(np.arange(len(dataset)), train_size=n,
                                           stratify=local_labels, random_state=seed)
    return Subset(dataset, sorted(map(int, subset_positions)))


def load_cub_loaders(cfg, include_test=False):
    """Load the public Hugging Face CUB train split; optionally the held-out test split."""
    from datasets import load_dataset
    raw_train = load_dataset('Multimodal-Fatima/CUB_train', split='train')
    raw_test = load_dataset('Multimodal-Fatima/CUB_test', split='test') if include_test else None
    if len(raw_train) != 5994 or (raw_test is not None and len(raw_test) != 5794):
        raise ValueError('Unexpected CUB-200 dataset version or length')
    if not {'image','label','file_name'}.issubset(raw_train.column_names):
        raise ValueError('Training dataset is missing required fields')
    labels = np.asarray(raw_train['label'], dtype=int)
    if set(labels) != set(range(200)):
        raise ValueError('Training labels must be integers 0..199')
    train_ids, val_ids, split_hash = get_or_create_split(raw_train, cfg)
    loader_kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], seed=cfg['seed'])
    loaders = {
        'train': make_loader(CUBClassificationDataset(raw_train, train_ids,
                 transforms_for(True, cfg['image_size'])), shuffle=True, **loader_kw),
        'validation': make_loader(CUBClassificationDataset(raw_train, val_ids,
                 transforms_for(False, cfg['image_size'])), shuffle=False, **loader_kw),
    }
    names = get_class_names(raw_train, raw_test)
    if len(names) != cfg['num_classes']:
        raise ValueError('Class count does not match config')
    if raw_test is not None:
        if not {'image','label','file_name'}.issubset(raw_test.column_names):
            raise ValueError('Test dataset missing required fields')
        if not set(map(int,raw_test['label'])).issubset(set(range(200))):
            raise ValueError('Test class indices do not match train class indices')
        loaders['test'] = make_loader(CUBClassificationDataset(raw_test,range(len(raw_test)),
                         transforms_for(False,cfg['image_size'])),shuffle=False,**loader_kw)
        loaders['test_filenames'] = list(raw_test['file_name'])
    return loaders, names, split_hash, labels
