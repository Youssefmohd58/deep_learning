"""Classification metrics on the held-out official test split, and aligned model comparison.

Metrics (explained in the README and report):
* accuracy        - fraction of test images whose top prediction is the true species.
* top5_accuracy   - true species is among the five highest-scoring classes.
* precision_macro - per species: of the images predicted as that species, how many were right; averaged
                    over the 200 species so every species counts equally.
* recall_macro    - per species: of that species' images, how many were found; averaged over species.
* f1_macro        - per-species harmonic mean of precision and recall, averaged over species.
* f1_weighted     - like f1_macro, but each species weighted by its number of test images.
* confusion matrix - counts of (true species, predicted species) pairs; shows which species get mixed up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


@torch.inference_mode()
def predict_loader(model, loader, device, file_names=None) -> pd.DataFrame:
    """One row per image: true label, prediction, confidence and whether top-5 contains the truth."""
    model.eval()
    rows = []
    for batch in loader:
        probs = model(batch['image'].to(device)).float().softmax(1)
        confidence, predicted = probs.max(1)
        top5 = probs.topk(5, dim=1).indices.cpu().tolist()
        for row, y, p, conf, k5 in zip(batch['row_index'].tolist(), batch['label'].tolist(),
                                       predicted.cpu().tolist(), confidence.cpu().tolist(), top5):
            item = {'row_index': int(row), 'y_true': int(y), 'y_pred': int(p), 'confidence': float(conf),
                    'top5_hit': int(int(y) in k5)}
            if file_names is not None:
                item['file_name'] = str(file_names[int(row)])
            rows.append(item)
    frame = pd.DataFrame(rows).sort_values('row_index').reset_index(drop=True)
    if frame['row_index'].duplicated().any():
        raise ValueError('Duplicate images in predictions')
    return frame


def metrics_for(predictions: pd.DataFrame, class_names) -> tuple[dict, np.ndarray, pd.DataFrame]:
    y = predictions['y_true'].to_numpy(dtype=int)
    p = predictions['y_pred'].to_numpy(dtype=int)
    k = len(class_names)
    if len(y) == 0 or y.min() < 0 or y.max() >= k or p.min() < 0 or p.max() >= k:
        raise ValueError('Class IDs out of range')
    labels = list(range(k))
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average='macro', zero_division=0)
    weighted_f1 = precision_recall_fscore_support(y, p, labels=labels, average='weighted', zero_division=0)[2]
    metrics = {'samples': int(len(y)), 'accuracy': float(accuracy_score(y, p)),
               'precision_macro': float(precision), 'recall_macro': float(recall),
               'f1_macro': float(f1), 'f1_weighted': float(weighted_f1)}
    if 'top5_hit' in predictions:
        metrics['top5_accuracy'] = float(predictions['top5_hit'].mean())
    cm = confusion_matrix(y, p, labels=labels)
    report = pd.DataFrame(classification_report(y, p, labels=labels, target_names=list(class_names),
                                                zero_division=0, output_dict=True)).T
    return metrics, cm, report


def most_confused_pairs(cm: np.ndarray, class_names, top=20) -> pd.DataFrame:
    """The (true, predicted) species pairs with the most mistakes, easier to read than a 200x200 matrix."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:top]
    rows = []
    for index in flat:
        t, p = np.unravel_index(index, off.shape)
        if off[t, p] == 0:
            break
        rows.append({'true_id': int(t), 'true_species': class_names[t], 'predicted_id': int(p),
                     'predicted_species': class_names[p], 'count': int(off[t, p]),
                     'fraction_of_true_class': float(off[t, p] / max(cm[t].sum(), 1))})
    return pd.DataFrame(rows)


def align_predictions(named_frames) -> list[tuple[str, pd.DataFrame]]:
    """Check every model predicted the same images with the same labels (and file names) before comparing."""
    if len(named_frames) < 2:
        raise ValueError('Comparison needs at least two models')
    base = named_frames[0][1].set_index('row_index').sort_index()
    aligned = []
    for name, frame in named_frames:
        frame = frame.set_index('row_index').sort_index()
        if not frame.index.equals(base.index):
            raise ValueError(f'{name} was evaluated on a different set of test images')
        if not np.array_equal(frame['y_true'].to_numpy(), base['y_true'].to_numpy()):
            raise ValueError(f'{name} has different reference labels')
        if 'file_name' in frame and 'file_name' in base and not np.array_equal(
                frame['file_name'].astype(str).to_numpy(), base['file_name'].astype(str).to_numpy()):
            raise ValueError(f'{name} has different image file names')
        aligned.append((name, frame))
    return aligned
