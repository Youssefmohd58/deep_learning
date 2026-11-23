"""EDA figures for Task 1.1 (spec section 5.9: sample images, class distribution, image statistics)."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.plots import save_figure


def plot_class_distribution(train_counts, test_counts, path):
    order = np.argsort(train_counts.to_numpy())
    fig, ax = plt.subplots(figsize=(16, 4))
    x = np.arange(len(train_counts))
    ax.bar(x, train_counts.to_numpy()[order], width=1.0, label='original train', color='tab:blue')
    ax.bar(x, -test_counts.reindex(train_counts.index).to_numpy()[order], width=1.0, label='official test',
           color='tab:orange')
    ax.axhline(train_counts.mean(), color='k', linestyle=':', linewidth=1, label=f'train mean {train_counts.mean():.1f}')
    ax.set(xlabel='Species (sorted by training count)', ylabel='Images (test shown below 0)', xlim=(-1, len(x)),
           title='Images per species: train (up) and test (down)')
    ax.set_xticks([])
    ax.legend()
    return save_figure(fig, path)


def plot_image_statistics(meta, flags, path):
    meta = meta[meta.decode_ok]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, column, label in ((axes[0], 'width', 'Width (px)'), (axes[1], 'height', 'Height (px)'),
                              (axes[2], 'aspect_ratio', 'Width / height')):
        for split, group in meta.groupby('split'):
            ax.hist(group[column], bins=40, alpha=0.6, label=split)
        if column != 'aspect_ratio':
            ax.axvline(224, color='red', linestyle='--', linewidth=1, label='224px model input')
        ax.set(xlabel=label, ylabel='Images')
        ax.legend(fontsize=8)
    axes[3].scatter(meta['mean_brightness'], meta['brightness_std'], s=3, alpha=0.3, label='all')
    axes[3].scatter(flags['mean_brightness'], flags['brightness_std'], s=6, color='red', label='flagged for review')
    axes[3].set(xlabel='Mean brightness (0-255)', ylabel='Brightness std (contrast)')
    axes[3].legend(fontsize=8)
    fig.suptitle('Image sizes and quality (train + test)')
    return save_figure(fig, path)


def plot_samples(ds, class_names, path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(ds), size=n, replace=False)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 4.6 * np.ceil(n / cols)))
    for ax, row in zip(axes.flat, rows):
        item = ds[int(row)]
        ax.imshow(item['image'].convert('RGB'))
        caption = (item.get('description') or '').splitlines()[0] if item.get('description') else ''
        ax.set_title(f'{class_names[int(item["label"])]}\n' + '\n'.join(textwrap.wrap(f'"{caption}"', 42)[:3]),
                     fontsize=8)
        ax.axis('off')
    for ax in list(axes.flat)[len(rows):]:
        ax.axis('off')
    fig.suptitle('Random training samples with their species and first caption')
    return save_figure(fig, path)


def plot_caption_statistics(captions, images, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(captions['words'], bins=range(0, int(captions['words'].max()) + 2), color='tab:green')
    axes[0].axvline(captions['words'].mean(), color='k', linestyle=':', label=f'mean {captions["words"].mean():.1f}')
    axes[0].set(xlabel='Words per caption (after cleaning)', ylabel='Captions', title='Caption length')
    axes[0].legend()
    counts = images['clean_captions'].value_counts().sort_index()
    axes[1].bar(counts.index.astype(str), counts.values, color='tab:purple')
    axes[1].set(xlabel='Captions per image after cleaning', ylabel='Images', title='Captions kept per image')
    return save_figure(fig, path)


def plot_top_words(top, path, title='Most frequent words (stop-words removed)'):
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(words) + 1))
    ax.barh(words[::-1], counts[::-1], color='tab:cyan')
    ax.set(xlabel='Occurrences in training captions', title=title)
    return save_figure(fig, path)
