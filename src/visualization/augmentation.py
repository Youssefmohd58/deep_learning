"""Augmentation visualisation (Task 1.1 deliverable): the same photos under each training recipe."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.data.transforms import build_transform
from src.visualization.plots import save_figure


def plot_augmentations(ds, rows, recipe, image_size, path, variants=5, seed=0):
    """Each row: the evaluation view of an image, then ``variants`` random training augmentations."""
    torch.manual_seed(seed)
    evaluation = build_transform(image_size, train=False, normalize=False)
    augment = build_transform(image_size, train=True, recipe=recipe, normalize=False)
    fig, axes = plt.subplots(len(rows), variants + 1, figsize=(2.3 * (variants + 1), 2.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        image = ds[int(row)]['image']
        views = [evaluation(image)] + [augment(image) for _ in range(variants)]
        for c, view in enumerate(views):
            axes[r, c].imshow(view.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title('evaluation (no aug.)' if c == 0 else f'augmented #{c}', fontsize=8)
    fig.suptitle(f'Training augmentation recipe "{recipe}" at {image_size}px')
    return save_figure(fig, path)
