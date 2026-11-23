"""Training engine and trainer tests on a tiny fake dataset (CPU, no downloads)."""
import torch
from torch import nn

from src.models.registry import build_model, load_checkpoint
from src.training.engine import run_epoch
from src.training.strategies import stages_for
from src.training.trainer import train_model
from src.utils.config import apply_overrides


def test_engine_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = build_model('scratch_cnn', num_classes=4, pretrained=False)
    batch = {'image': torch.rand(8, 3, 32, 32), 'label': torch.arange(8) % 4, 'row_index': torch.arange(8)}
    loader = [batch]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    first = run_epoch(model, loader, device, criterion, optimizer)['loss']
    for _ in range(15):
        last = run_epoch(model, loader, device, criterion, optimizer)['loss']
    assert last < first


def test_transfer_strategy_has_head_then_finetune(tiny_cfg):
    cfg = apply_overrides(tiny_cfg, ['training.strategy=transfer', 'training.head_epochs=2',
                                     'training.finetune_epochs=0', 'training.head_lr=0.001',
                                     'training.finetune_lr=0.0001'])
    assert [s.name for s in stages_for(cfg)] == ['head']


def test_train_model_saves_unified_checkpoint(tiny_cfg, fake_hf):
    cfg = apply_overrides(tiny_cfg, ['training.epochs=2'])
    summary = train_model(cfg, data={'train': fake_hf}, verbose=False)
    assert summary['epochs_run'] == 2 and 0 <= summary['best_val_accuracy'] <= 1
    loaded = load_checkpoint(summary['checkpoint'])
    assert loaded.architecture == 'scratch_cnn' and loaded.image_size == 32
    assert loaded.meta['split_sha256'] == summary['split_sha256'] and len(loaded.class_names) == 4


def test_train_transfer_two_stages(tiny_cfg, fake_hf):
    cfg = apply_overrides(tiny_cfg, ['model.architecture=resnet18', 'model.pretrained=false',
                                     'training.strategy=transfer', 'training.head_epochs=1',
                                     'training.finetune_epochs=1', 'training.head_lr=0.001',
                                     'training.finetune_lr=0.0001', 'training.checkpoint_name=r.pt',
                                     'data.augmentation=transfer_mild'])
    summary = train_model(cfg, data={'train': fake_hf}, verbose=False)
    assert list(summary['history']['stage']) == ['head', 'finetune']
