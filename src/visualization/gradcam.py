"""Grad-CAM (spec section 5.6): where in the image a model looked for a given class.

Grad-CAM takes the last convolutional block's activations, weights each channel by the
average gradient of the class score with respect to it, sums them and keeps the positive
part. Bright regions are the image areas that most increased that class's score.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from src.data.transforms import denormalize
from src.models.registry import cam_layer
from src.visualization.plots import save_figure


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activation = None
        self.gradient = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activation = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradient = gradient

    def compute(self, image_tensor, target_class) -> 'np.ndarray':
        """Heatmap in [0, 1] with the same height/width as the input image."""
        self.model.eval()
        device = next(self.model.parameters()).device
        x = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, int(target_class)].backward()
        if self.gradient is None:
            raise RuntimeError('Grad-CAM did not receive gradients from the target layer')
        weights = self.gradient.float().mean(dim=(2, 3), keepdim=True)
        heat = (weights * self.activation.float()).sum(dim=1, keepdim=True).relu()
        heat = F.interpolate(heat, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        heat = heat.detach().cpu().numpy()
        heat -= heat.min()
        return heat / max(float(heat.max()), 1e-8)

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _overlay(ax, image, heat, title):
    ax.imshow(image)
    if heat is not None:
        ax.imshow(heat, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def explain_image(loaded, image_tensor, target_class=None) -> tuple[int, 'np.ndarray']:
    """Grad-CAM for one preprocessed image; explains the predicted class unless one is given."""
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        predicted = int(loaded.model(image_tensor.unsqueeze(0).to(device)).argmax(1))
    target = predicted if target_class is None else int(target_class)
    with GradCAM(loaded.model, cam_layer(loaded.model, loaded.architecture)) as cam:
        return target, cam.compute(image_tensor, target)


def plot_single(loaded, image_tensor, path, target_class=None) -> Path:
    target, heat = explain_image(loaded, image_tensor, target_class)
    image = denormalize(image_tensor)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _overlay(axes[0], image, None, 'Input image')
    _overlay(axes[1], image, heat, f'Grad-CAM: {loaded.class_names[target]}')
    fig.suptitle(loaded.display_name)
    return save_figure(fig, path)


def plot_cases(loaded, predictions, dataset, path, per_kind=4) -> Path:
    """Grid of the most confident CORRECT and the most confident INCORRECT test predictions.

    Each row: the image, Grad-CAM for the predicted class and Grad-CAM for the true class.
    For a wrong prediction the two maps show what misled the model versus what it should have used.
    """
    position = {int(r): i for i, r in enumerate(dataset.indices)}
    correct = predictions[predictions.y_true == predictions.y_pred].nlargest(per_kind, 'confidence')
    wrong = predictions[predictions.y_true != predictions.y_pred].nlargest(per_kind, 'confidence')
    cases = [(row, True) for row in correct.itertuples()] + [(row, False) for row in wrong.itertuples()]
    if not cases:
        raise ValueError('No predictions to explain')
    names = loaded.class_names
    fig, axes = plt.subplots(len(cases), 3, figsize=(11, 3.4 * len(cases)), squeeze=False)
    with GradCAM(loaded.model, cam_layer(loaded.model, loaded.architecture)) as cam:
        for i, (row, ok) in enumerate(cases):
            x = dataset[position[int(row.row_index)]]['image']
            image = denormalize(x)
            label = 'CORRECT' if ok else 'WRONG'
            _overlay(axes[i, 0], image, None, f'{label} (conf {row.confidence:.0%})\ntrue: {names[row.y_true]}')
            _overlay(axes[i, 1], image, cam.compute(x, row.y_pred), f'Grad-CAM for predicted:\n{names[row.y_pred]}')
            if ok:
                _overlay(axes[i, 2], image, None, 'Prediction = true class')
            else:
                _overlay(axes[i, 2], image, cam.compute(x, row.y_true), f'Grad-CAM for true class:\n{names[row.y_true]}')
    fig.suptitle(f'{loaded.display_name}: Grad-CAM on correct and incorrect test predictions', fontsize=11)
    return save_figure(fig, path)
