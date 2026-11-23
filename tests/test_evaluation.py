"""Metrics, alignment and Grad-CAM tests (no downloads)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import CUBClassificationDataset
from src.data.transforms import build_transform
from src.evaluation.classification import align_predictions, metrics_for, most_confused_pairs
from src.models.registry import LoadedModel, build_model
from src.visualization.gradcam import GradCAM, plot_cases
from src.models.registry import cam_layer

NAMES = [f'Class {i}' for i in range(200)]


def test_metrics_and_confused_pairs():
    preds = pd.DataFrame({'row_index': [0, 1, 2, 3], 'y_true': [0, 1, 1, 2],
                          'y_pred': [0, 2, 2, 2], 'top5_hit': [1, 1, 1, 1]})
    metrics, cm, report = metrics_for(preds, NAMES)
    assert np.isclose(metrics['accuracy'], 0.5) and metrics['top5_accuracy'] == 1.0
    assert cm.shape == (200, 200)
    pairs = most_confused_pairs(cm, NAMES)
    assert pairs.iloc[0][['true_id', 'predicted_id', 'count']].tolist() == [1, 2, 2]


def test_alignment_rejects_different_images_or_labels():
    a = pd.DataFrame({'row_index': [0, 1], 'y_true': [0, 1], 'y_pred': [0, 1], 'file_name': ['a', 'b']})
    b = a.sample(frac=1, random_state=0)
    assert len(align_predictions([('a', a), ('b', b)])) == 2
    c = a.assign(file_name=['a', 'z'])
    with pytest.raises(ValueError, match='file names'):
        align_predictions([('a', a), ('c', c)])


@pytest.mark.parametrize('arch', ['scratch_cnn', 'resnet18', 'efficientnet_b0'])
def test_gradcam_heatmap_matches_input(arch):
    torch.set_num_threads(2)
    model = build_model(arch, 200, pretrained=False).eval()
    with GradCAM(model, cam_layer(model, arch)) as cam:
        heat = cam.compute(torch.rand(3, 64, 64), 5)
    assert heat.shape == (64, 64) and 0 <= heat.min() and heat.max() <= 1


def test_gradcam_case_grid(tmp_path, fake_hf):
    model = build_model('scratch_cnn', 4, pretrained=False).eval()
    loaded = LoadedModel(model, 'scratch_cnn', ['a', 'b', 'c', 'd'], 32, 'scratch_cnn')
    ds = CUBClassificationDataset(fake_hf, range(6), build_transform(32))
    preds = pd.DataFrame({'row_index': range(6), 'y_true': [0, 1, 2, 3, 0, 1], 'y_pred': [0, 1, 0, 0, 1, 1],
                          'confidence': np.linspace(0.2, 0.9, 6)})
    path = plot_cases(loaded, preds, ds, tmp_path / 'g.png', per_kind=2)
    assert path.is_file()
