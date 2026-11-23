"""Image preprocessing and named augmentation recipes (Task 1.1).

One input size for all models (``data.image_size``, 224 by default). Each model config
picks a training augmentation *recipe* by name; evaluation preprocessing is the same,
deterministic pipeline for every model. Rationale: docs/preprocessing_decisions.md.
"""
from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 256 / 224  # resize the shorter side to 256 for a 224 crop (standard ImageNet practice)


def _to_rgb(image):
    return image.convert('RGB')


def _training_steps(recipe: str, size: int) -> tuple[list, list]:
    """Return (PIL-image steps, tensor steps) for a training recipe."""
    if recipe == 'scratch_strong':
        # Large crops/zoom, flips and random photometric ops, plus random erasing (occlusion).
        # A model with no prior knowledge overfits 4,795 images quickly without this.
        pil = [T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.33),
                                   interpolation=InterpolationMode.BILINEAR),
               T.RandomHorizontalFlip(p=0.5),
               T.TrivialAugmentWide()]
        tensor = [T.RandomErasing(p=0.25)]
    elif recipe == 'transfer_mild':
        # Gentle changes that keep birds recognisable; pretrained features need little help.
        pil = [T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15),
                                   interpolation=InterpolationMode.BILINEAR),
               T.RandomHorizontalFlip(p=0.5),
               T.RandomRotation(10),
               T.ColorJitter(brightness=0.1, contrast=0.1)]
        tensor = []
    elif recipe == 'none':
        return _eval_steps(size), []
    else:
        raise ValueError(f'Unknown augmentation recipe {recipe!r}; choose from {RECIPES}')
    return pil, tensor


def _eval_steps(size: int) -> list:
    return [T.Resize(round(size * RESIZE_RATIO), interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(size)]


RECIPES = ('scratch_strong', 'transfer_mild', 'none')


def build_transform(image_size: int = 224, train: bool = False, recipe: str = 'transfer_mild',
                    normalize: bool = True):
    """Full transform: PIL image -> normalised float tensor of shape (3, size, size).

    ``normalize=False`` returns un-normalised tensors in [0, 1] for visualising augmentation.
    """
    size = int(image_size)
    pil, tensor = _training_steps(recipe, size) if train else (_eval_steps(size), [])
    steps = [T.Lambda(_to_rgb), *pil, T.ToTensor()]
    if normalize:
        steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    steps += tensor
    return T.Compose(steps)


def denormalize(tensor):
    """Undo ImageNet normalisation; returns an HxWx3 NumPy image in [0, 1] for plotting."""
    import numpy as np

    mean = np.asarray(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD).reshape(1, 1, 3)
    image = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * std + mean, 0, 1)
