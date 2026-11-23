"""One place to build any model and to save/load any checkpoint.

The rest of the code never needs to know which architecture it is handling, so models
can be swapped without touching training, evaluation or the CLI (spec section 5.2).

Checkpoint formats understood by ``load_checkpoint``:
* unified (written by this project): architecture, model_state_dict, class_names, image_size, ...
* Member 3 (earlier Colab runs): architecture, class_names, model_state_dict, config{image_size}
* Member 2 legacy: model_state_dict, num_classes, epoch, val_accuracy -> scratch_cnn trained at 128px
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.models.bird_cnn import BirdCNN
from src.models.transfer import (TRANSFER_ARCHITECTURES, build_transfer_model, set_trainable,
                                 transfer_cam_layer, transfer_features)

SCRATCH = 'scratch_cnn'
ARCHITECTURES = (SCRATCH, *TRANSFER_ARCHITECTURES)
LEGACY_SCRATCH_IMAGE_SIZE = 128  # Member 2's original checkpoint was trained on 128x128 inputs


def build_model(architecture, num_classes=200, pretrained=True, dropout=0.5):
    if architecture == SCRATCH:
        return BirdCNN(num_classes=num_classes, dropout=dropout)
    return build_transfer_model(architecture, num_classes, pretrained)


def is_transfer(architecture) -> bool:
    return architecture in TRANSFER_ARCHITECTURES


def prepare_stage(model, architecture, stage):
    """Set which parameters train in a stage ('full' for scratch, 'head'/'finetune' for transfer)."""
    if architecture == SCRATCH:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        set_trainable(model, architecture, stage)


def cam_layer(model, architecture):
    """Last convolutional block: the layer Grad-CAM explains."""
    return model.encoder[-1] if architecture == SCRATCH else transfer_cam_layer(model, architecture)


def encode(model, architecture, images):
    """Encoder embedding for downstream tasks (Milestone 2 colorization/captioning)."""
    return model.features(images) if architecture == SCRATCH else transfer_features(model, architecture, images)


def parameter_count(model, trainable_only=False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    architecture: str
    class_names: list
    image_size: int
    tag: str                      # unique name for result files, e.g. scratch_cnn_legacy128
    meta: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return {'scratch_cnn': 'Scratch CNN', 'resnet18': 'ResNet-18',
                'efficientnet_b0': 'EfficientNet-B0'}[self.architecture] + f' ({self.image_size}px)'


def save_checkpoint(path, model, architecture, class_names, image_size, **meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'cub200-unified-v1', 'architecture': architecture,
               'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'class_names': list(class_names), 'image_size': int(image_size), **meta}
    torch.save(payload, path)
    return path


def checkpoint_architecture(payload: dict) -> str:
    if 'architecture' in payload:
        return str(payload['architecture'])
    state = payload.get('model_state_dict', {})
    if any(str(k).startswith('encoder.') for k in state):
        return SCRATCH
    raise ValueError('Unknown checkpoint structure: no architecture and not a BirdCNN state dict')


def load_checkpoint(path, device='cpu', fallback_class_names=None) -> LoadedModel:
    """Load any supported checkpoint safely (tensors only, no pickled code) and in eval mode."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'model_state_dict' not in payload:
        raise ValueError(f'{path} is not a model checkpoint with a model_state_dict')
    architecture = checkpoint_architecture(payload)
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unsupported architecture {architecture!r} in {path}')

    legacy = 'format' not in payload
    names = payload.get('class_names') or fallback_class_names
    if not names:
        raise ValueError(f'{path} has no class names; pass fallback_class_names (configs/class_names.json)')
    if 'image_size' in payload:
        size = int(payload['image_size'])
    elif isinstance(payload.get('config'), dict) and 'image_size' in payload['config']:
        size = int(payload['config']['image_size'])
    elif architecture == SCRATCH:
        size = LEGACY_SCRATCH_IMAGE_SIZE
    else:
        size = 224

    model = build_model(architecture, num_classes=len(names), pretrained=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device).eval()

    tag = architecture
    if architecture == SCRATCH and legacy and size == LEGACY_SCRATCH_IMAGE_SIZE:
        tag = 'scratch_cnn_legacy128'
    meta = {k: v for k, v in payload.items() if k not in ('model_state_dict', 'class_names')}
    meta['path'] = str(path)
    meta['legacy_format'] = legacy
    meta.setdefault('best_val_accuracy', _fraction(payload.get('best_val_accuracy', payload.get('val_accuracy'))))
    return LoadedModel(model, architecture, list(names), size, tag, meta)


def _fraction(value):
    """Validation accuracy as a 0-1 fraction (Member 2 saved percentages)."""
    if value is None:
        return None
    value = float(value)
    return value / 100 if value > 1 else value
