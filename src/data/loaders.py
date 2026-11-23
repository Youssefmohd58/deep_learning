"""Reusable DataLoaders (Task 1.1 deliverable) built from a config.

    data = load_cub(cfg)
    loaders = build_loaders(cfg, data)               # train / validation (+ test if loaded)
    loaders = build_loaders(cfg, data, image_size=128) # e.g. for the legacy 128px checkpoint
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CUBClassificationDataset
from src.data.split import load_split, stratified_subset
from src.data.transforms import build_transform


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, shuffle, num_workers=0, seed=42) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=int(num_workers), generator=generator,
                      worker_init_fn=_seed_worker, persistent_workers=int(num_workers) > 0,
                      pin_memory=torch.cuda.is_available())


@dataclass
class Loaders:
    train: DataLoader | None = None
    validation: DataLoader | None = None
    test: DataLoader | None = None
    split_sha256: str = ''
    image_size: int = 224
    meta: dict = field(default_factory=dict)


def build_loaders(cfg: dict, data: dict, image_size=None, augmentation=None,
                  batch_size=None, include=('train', 'validation', 'test')) -> Loaders:
    """Build loaders from already-loaded HF datasets (see src.data.hf.load_cub)."""
    d = cfg['data']
    size = int(image_size or d['image_size'])
    recipe = augmentation or d.get('augmentation', 'transfer_mild')
    batch = int(batch_size or d['batch_size'])
    workers, seed = int(d.get('num_workers', 0)), int(cfg.get('seed', 42))
    out = Loaders(image_size=size)
    eval_tf = build_transform(size, train=False)

    if {'train', 'validation'} & set(include):
        train_hf = data['train']
        labels = np.asarray(train_hf['label'], dtype=int)
        train_ids, val_ids, out.split_sha256 = load_split(cfg, labels)
        train_ids = stratified_subset(train_ids, labels, d.get('max_train_images'), seed)
        val_ids = stratified_subset(val_ids, labels, d.get('max_val_images'), seed)
        if 'train' in include:
            ds = CUBClassificationDataset(train_hf, train_ids, build_transform(size, True, recipe))
            out.train = make_loader(ds, batch, True, workers, seed)
        if 'validation' in include:
            out.validation = make_loader(CUBClassificationDataset(train_hf, val_ids, eval_tf),
                                         batch, False, workers, seed)
        out.meta.update(train_images=len(train_ids), val_images=len(val_ids), augmentation=recipe)

    if 'test' in include and data.get('test') is not None:
        test_ds = CUBClassificationDataset(data['test'], None, eval_tf)
        out.test = make_loader(test_ds, batch, False, workers, seed)
        out.meta['test_images'] = len(test_ds)
    return out


def verify_loader(loader, expected_images, image_size, num_classes=200) -> dict:
    """Iterate a whole loader: shapes, finite values, label range, no missing/repeated rows."""
    total, seen = 0, []
    for batch in loader:
        x, y = batch['image'], batch['label']
        assert x.ndim == 4 and tuple(x.shape[1:]) == (3, image_size, image_size), f'bad shape {tuple(x.shape)}'
        assert torch.isfinite(x).all(), 'non-finite pixel values'
        assert bool(((y >= 0) & (y < num_classes)).all()), 'label out of range'
        total += len(y)
        seen.extend(batch['row_index'].tolist())
    assert total == expected_images and len(seen) == len(set(seen)), 'incomplete or repeated rows'
    return {'passed': True, 'images': total, 'batches': len(loader)}
