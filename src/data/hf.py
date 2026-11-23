"""Load CUB-200-2011 from Hugging Face and read its class names.

Spec section 3.1: ``Multimodal-Fatima/CUB_train`` (5,994 images) and
``Multimodal-Fatima/CUB_test`` (5,794 images); every record has image, label,
description (ten captions), file_name and id. Nothing downloads at import time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.config import project_path

EXPECTED_SIZES = {'train': 5994, 'test': 5794}


def load_cub(cfg: dict, splits=('train', 'test')) -> dict:
    """Return {'train': Dataset, 'test': Dataset} for the requested official splits."""
    from datasets import load_dataset

    data = cfg['data']
    names = {'train': data['train_dataset'], 'test': data['test_dataset']}
    result = {}
    for split in splits:
        ds = load_dataset(names[split], split=split)
        if len(ds) != EXPECTED_SIZES[split]:
            raise ValueError(f'Unexpected CUB {split} size {len(ds)} (expected {EXPECTED_SIZES[split]})')
        missing = {'image', 'label', 'file_name'} - set(ds.column_names)
        if missing:
            raise ValueError(f'CUB {split} split is missing columns {missing}')
        result[split] = ds
    return result


def dataset_class_names(train_ds, test_ds=None) -> list[str]:
    """Class names in label-ID order, from the ClassLabel feature or, failing that, file names."""
    feature = getattr(train_ds, 'features', {}).get('label') if hasattr(train_ds, 'features') else None
    names = list(getattr(feature, 'names', []) or [])
    if not names:
        by_id = {}
        for label, file_name in zip(train_ds['label'], train_ds['file_name']):
            species = str(file_name).rsplit('_', 2)[0].replace('_', ' ')
            if by_id.setdefault(int(label), species) != species:
                raise ValueError(f'Inconsistent species name for class {label}')
        names = [by_id[i] for i in range(len(by_id))]
    if test_ds is not None and hasattr(test_ds, 'features'):
        test_names = list(getattr(test_ds.features.get('label'), 'names', []) or [])
        if test_names and [normalize_name(n) for n in test_names] != [normalize_name(n) for n in names]:
            raise ValueError('Train and test class names differ')
    return names


def normalize_name(name: str) -> str:
    """Compare species names ignoring case, underscores and repeated spaces."""
    return re.sub(r'\s+', ' ', str(name).replace('_', ' ')).strip().casefold()


def same_class_order(a, b) -> bool:
    return len(a) == len(b) and all(normalize_name(x) == normalize_name(y) for x, y in zip(a, b))


def load_class_names(cfg: dict) -> list[str]:
    """Saved class names (configs/class_names.json) so inference works without the dataset."""
    path = project_path(cfg['data']['class_names_file'])
    names = json.loads(Path(path).read_text(encoding='utf-8'))
    if len(names) != int(cfg['data']['num_classes']):
        raise ValueError(f'{path} must list {cfg["data"]["num_classes"]} class names')
    return names


def save_class_names(cfg: dict, names) -> Path:
    path = project_path(cfg['data']['class_names_file'])
    path.write_text(json.dumps(list(names), indent=2), encoding='utf-8')
    return path
