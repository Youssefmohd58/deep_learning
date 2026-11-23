"""Data pipeline tests: split rules, transforms, dataset and loaders (no downloads)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import CUBClassificationDataset
from src.data.split import create_split, file_sha256, stratified_subset, validate_manifest
from src.data.transforms import RECIPES, build_transform
from src.utils.config import load_config, project_path


def test_team_split_is_valid_and_unchanged():
    cfg = load_config('configs/data.yaml')
    path = project_path(cfg['data']['split_manifest'])
    manifest = pd.read_csv(path)
    labels = np.zeros(5994, dtype=int)
    labels[manifest.row_index.to_numpy()] = manifest.label.to_numpy()
    validate_manifest(manifest, labels)
    assert (manifest.split == 'train').sum() == 4795 and (manifest.split == 'validation').sum() == 1199
    # The checkpoints from Members 2 and 3 were trained with exactly this file.
    assert file_sha256(path) == '4da0624b7b8bdc0a1a54aaf039e7b12fc029652ef158fb1a20c1c762288d3469'


def test_create_split_never_overwrites(tmp_path):
    labels = np.arange(100) % 4
    path = tmp_path / 's.csv'
    frame = create_split(labels, path, 0.25, seed=1)
    assert set(frame.split) == {'train', 'validation'}
    with pytest.raises(FileExistsError):
        create_split(labels, path)


def test_manifest_with_wrong_labels_is_rejected(tmp_path):
    labels = np.arange(40) % 4
    frame = create_split(labels, tmp_path / 's.csv', 0.25)
    frame.loc[0, 'label'] = 3 if frame.loc[0, 'label'] != 3 else 2
    with pytest.raises(ValueError, match='labels differ'):
        validate_manifest(frame, labels, num_classes=4)


@pytest.mark.parametrize('recipe', RECIPES)
def test_every_recipe_outputs_224(fake_hf, recipe):
    x = build_transform(224, train=True, recipe=recipe)(fake_hf[0]['image'])
    assert x.shape == (3, 224, 224) and torch.isfinite(x).all()
    assert build_transform(224, train=False)(fake_hf[0]['image']).shape == (3, 224, 224)


def test_dataset_returns_original_row_index(fake_hf):
    ds = CUBClassificationDataset(fake_hf, [5, 9], build_transform(32))
    item = ds[1]
    assert int(item['row_index']) == 9 and int(item['label']) == fake_hf.labels[9]


def test_stratified_subset_keeps_every_class():
    labels = np.arange(400) % 4
    subset = stratified_subset(range(400), labels, 40, seed=0)
    assert len(subset) == 40 and set(labels[subset]) == {0, 1, 2, 3}
