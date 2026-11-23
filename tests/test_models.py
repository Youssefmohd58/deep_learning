"""Model and checkpoint registry tests."""
import pytest
import torch

from src.models.registry import (ARCHITECTURES, build_model, cam_layer, encode, load_checkpoint,
                                 parameter_count, prepare_stage, save_checkpoint)
from src.utils.config import project_path

NAMES = [f'Species {i}' for i in range(200)]


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_build_and_forward_224(arch):
    torch.set_num_threads(2)
    model = build_model(arch, 200, pretrained=False).eval()
    with torch.inference_mode():
        x = torch.rand(2, 3, 224, 224)
        assert model(x).shape == (2, 200)
        assert encode(model, arch, x).ndim == 2
    assert cam_layer(model, arch) is not None


def test_transfer_stages_unfreeze_more_in_finetune():
    model = build_model('resnet18', pretrained=False)
    prepare_stage(model, 'resnet18', 'head')
    head = parameter_count(model, trainable_only=True)
    prepare_stage(model, 'resnet18', 'finetune')
    assert parameter_count(model, trainable_only=True) > head > 0


@pytest.mark.parametrize('arch', ARCHITECTURES)
def test_unified_checkpoint_roundtrip(tmp_path, arch):
    model = build_model(arch, pretrained=False)
    path = save_checkpoint(tmp_path / 'm.pt', model, arch, NAMES, 224, epoch=3, best_val_accuracy=0.5)
    loaded = load_checkpoint(path)
    assert loaded.architecture == arch and loaded.image_size == 224 and loaded.tag == arch
    assert loaded.class_names[7] == 'Species 7' and loaded.meta['epoch'] == 3


def test_legacy_member2_format_is_128px(tmp_path):
    model = build_model('scratch_cnn')
    torch.save({'model_state_dict': model.state_dict(), 'num_classes': 200, 'epoch': 78,
                'val_accuracy': 60.05}, tmp_path / 'legacy.pth')
    loaded = load_checkpoint(tmp_path / 'legacy.pth', fallback_class_names=NAMES)
    assert loaded.image_size == 128 and loaded.tag == 'scratch_cnn_legacy128'
    assert abs(loaded.meta['best_val_accuracy'] - 0.6005) < 1e-6


REAL = {'scratch_cnn_legacy128.pth': ('scratch_cnn', 128),
        'resnet18_best.pt': ('resnet18', 224),
        'efficientnet_b0_best.pt': ('efficientnet_b0', 224)}


@pytest.mark.parametrize('name', sorted(REAL))
def test_real_team_checkpoints_load_strictly(name):
    path = project_path('checkpoints') / name
    if not path.is_file():
        pytest.skip(f'{name} not downloaded (see README: checkpoints from Google Drive)')
    loaded = load_checkpoint(path, fallback_class_names=NAMES)
    assert (loaded.architecture, loaded.image_size) == REAL[name]
    assert len(loaded.class_names) == 200
