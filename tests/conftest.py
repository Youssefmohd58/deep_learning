"""Shared test fixtures: a tiny in-memory stand-in for the Hugging Face CUB dataset."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeHF:
    """Mimics the parts of datasets.Dataset used by the project (row and column access)."""

    def __init__(self, n=40, classes=4, size=(64, 48), seed=0):
        rng = np.random.default_rng(seed)
        self.labels = [i % classes for i in range(n)]
        self.images = [Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
                       for _ in range(n)]
        self.captions = ['a small bird with a red crown.\nthis bird has white wings.\nthis'] * n
        self.column_names = ['image', 'label', 'description', 'file_name', 'id']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if isinstance(key, str):
            return {'label': self.labels, 'file_name': [f'Bird_{l}_{i:04d}_1.jpg' for i, l in enumerate(self.labels)],
                    'description': self.captions, 'id': list(range(len(self))), 'image': self.images}[key]
        return {'image': self.images[key], 'label': self.labels[key], 'description': self.captions[key],
                'file_name': f'Bird_{self.labels[key]}_{key:04d}_1.jpg', 'id': key}

    def __iter__(self):
        return (self[i] for i in range(len(self)))


@pytest.fixture
def fake_hf():
    return FakeHF()


@pytest.fixture
def tiny_cfg(tmp_path, fake_hf):
    """Config pointing at a temporary 4-class split for the fake dataset."""
    from src.data.split import create_split
    from src.utils.config import load_config

    manifest = tmp_path / 'split.csv'
    create_split(fake_hf.labels, manifest, validation_fraction=0.25, seed=0)
    cfg = load_config('configs/scratch_cnn.yaml', [
        f'data.split_manifest={manifest.as_posix()}', 'data.image_size=32', 'data.batch_size=8',
        f'project.output_dir={(tmp_path / "out").as_posix()}',
        f'project.checkpoint_dir={(tmp_path / "ckpt").as_posix()}',
        'tracking.enabled=false'])
    return cfg
