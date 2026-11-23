"""The fixed, stratified train/validation split (Member 1, Task 1.1).

The official test split is never touched for model selection. The original 5,994
training images are split 80/20, stratified by species, and the result is saved as a
manifest CSV. Every model reads the same manifest, and its SHA-256 is stored inside
each checkpoint so comparisons can prove they used the same split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import project_path

REQUIRED_COLUMNS = {'row_index', 'split', 'label'}


def manifest_path(cfg: dict) -> Path:
    return project_path(cfg['data']['split_manifest'])


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_split(labels, path, validation_fraction=0.2, seed=42, file_names=None) -> pd.DataFrame:
    """Create a new stratified manifest. Refuses to overwrite an existing one."""
    from sklearn.model_selection import train_test_split

    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} already exists; the team split is never overwritten')
    labels = np.asarray(labels, dtype=int)
    rows = np.arange(len(labels))
    _, val_rows = train_test_split(rows, test_size=validation_fraction, stratify=labels, random_state=seed)
    split = np.full(len(rows), 'train', dtype=object)
    split[val_rows] = 'validation'
    frame = pd.DataFrame({'row_index': rows, 'split': split, 'label': labels})
    if file_names is not None:
        frame['file_name'] = list(file_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def validate_manifest(manifest: pd.DataFrame, labels, num_classes=200) -> None:
    """Every original training row appears once, labels match the dataset, both splits cover all classes."""
    labels = np.asarray(labels, dtype=int)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f'Split manifest is missing columns {missing}')
    if len(manifest) != len(labels) or manifest['row_index'].duplicated().any():
        raise ValueError('Split manifest rows are missing or duplicated')
    rows = manifest['row_index'].astype(int).to_numpy()
    if set(rows) != set(range(len(labels))):
        raise ValueError('Split manifest row indices do not match the training set')
    if set(manifest['split']) != {'train', 'validation'}:
        raise ValueError("Split manifest may only contain 'train' and 'validation'")
    if not np.array_equal(manifest['label'].to_numpy(dtype=int), labels[rows]):
        raise ValueError('Split manifest labels differ from the dataset: wrong dataset version?')
    for name in ('train', 'validation'):
        ids = rows[manifest['split'].to_numpy() == name]
        if len(set(labels[ids])) != num_classes:
            raise ValueError(f'The {name} split does not cover all {num_classes} classes')


def load_split(cfg: dict, train_labels) -> tuple[list[int], list[int], str]:
    """Return (train row ids, validation row ids, manifest sha256) after validating the manifest."""
    path = manifest_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f'Split manifest not found: {path}. Restore it from git, or create a new one with '
            '`python -m src.cli eda --create-split` (this changes results, so agree it with the team).')
    manifest = pd.read_csv(path)
    validate_manifest(manifest, train_labels, len(set(np.asarray(train_labels).tolist())))
    train_ids = manifest.loc[manifest['split'] == 'train', 'row_index'].astype(int).tolist()
    val_ids = manifest.loc[manifest['split'] == 'validation', 'row_index'].astype(int).tolist()
    return train_ids, val_ids, file_sha256(path)


def stratified_subset(ids, labels, n, seed=42) -> list[int]:
    """Smaller class-balanced subset of row ids, used for smoke tests and quick tuning."""
    ids = list(ids)
    if n is None or n >= len(ids):
        return ids
    labels = np.asarray(labels, dtype=int)
    local = labels[np.asarray(ids, dtype=int)]
    classes = len(set(local.tolist()))
    if n < classes:
        rng = np.random.default_rng(seed)
        return sorted(int(i) for i in rng.choice(ids, size=n, replace=False))
    from sklearn.model_selection import train_test_split

    chosen, _ = train_test_split(np.asarray(ids), train_size=n, stratify=local, random_state=seed)
    return sorted(int(i) for i in chosen)
