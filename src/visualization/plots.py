"""Training curves, confusion matrices, model comparison and tuning plots (Matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # files only; works on servers and in tests without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def save_figure(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_history(history, path, title='Training vs validation') -> Path:
    """Loss and accuracy per epoch; dotted lines mark where a new training stage starts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in ((axes[0], 'loss', 'Cross-entropy loss'), (axes[1], 'accuracy', 'Accuracy')):
        ax.plot(history['epoch'], history[f'train_{metric}'], label='train')
        ax.plot(history['epoch'], history[f'val_{metric}'], label='validation', linestyle='--')
        ax.set(xlabel='Epoch', ylabel=label)
        ax.grid(alpha=0.25)
        if 'stage' in history and history['stage'].nunique() > 1:
            for start in history.groupby('stage', sort=False)['epoch'].min().iloc[1:]:
                ax.axvline(start - 0.5, color='grey', linestyle=':', label='stage change')
        ax.legend()
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.suptitle(title)
    return save_figure(fig, path)


def plot_confusion(cm, path, title) -> Path:
    """Row-normalised 200x200 confusion matrix (each row sums to 1 = recall per class)."""
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set(xlabel='Predicted class ID', ylabel='True class ID', title=title)
    fig.colorbar(image, ax=ax, shrink=0.8, label='Fraction of true class')
    return save_figure(fig, path)


def plot_model_comparison(table, path, metrics=('accuracy', 'f1_macro', 'top5_accuracy')) -> Path:
    """Grouped bars: one group per model, one bar per metric."""
    metrics = [m for m in metrics if m in table.columns]
    x = np.arange(len(table))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(table)), 4.5))
    for i, metric in enumerate(metrics):
        values = table[metric].to_numpy(dtype=float) * 100
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * width, values, width, label=metric.replace('_', ' '))
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    ax.set_xticks(x, table['model'])
    ax.set(ylabel='Score on official test set (%)', ylim=(0, 105), title='CUB-200-2011: scratch vs transfer learning')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    return save_figure(fig, path)


def plot_tuning(trials, path, metric='best_val_accuracy') -> Path:
    """Left: every trial's validation accuracy by method. Right: best score vs total time per method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, group in trials.groupby('method'):
        axes[0].plot(group['trial'], group[metric] * 100, marker='o', label=method)
    axes[0].set(xlabel='Trial', ylabel='Best validation accuracy (%)', title='Every trial')
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    summary = trials.groupby('method').agg(best=(metric, 'max'), seconds=('duration_seconds', 'sum'),
                                           trials=('trial', 'count'))
    for method, row in summary.iterrows():
        axes[1].scatter(row['seconds'] / 60, row['best'] * 100, s=80)
        axes[1].annotate(f'{method} ({int(row["trials"])} trials)', (row['seconds'] / 60, row['best'] * 100),
                         textcoords='offset points', xytext=(6, 6))
    axes[1].set(xlabel='Total tuning time (minutes)', ylabel='Best validation accuracy (%)',
                title='Grid vs random search')
    axes[1].grid(alpha=0.25)
    return save_figure(fig, path)


def plot_benchmark(table, path) -> Path:
    """Horizontal bars of top-1 accuracy: our models highlighted next to published/open-source results."""
    table = table.sort_values('accuracy')
    colors = ['tab:blue' if src == 'this project' else 'tab:grey' for src in table['source']]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
    bars = ax.barh(table['model'], table['accuracy'] * 100, color=colors)
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    ax.set(xlabel='Top-1 accuracy on CUB-200-2011 test set (%)', xlim=(0, 100),
           title='Benchmark: this project (blue) vs published / open-source (grey)')
    ax.grid(axis='x', alpha=0.25)
    return save_figure(fig, path)
