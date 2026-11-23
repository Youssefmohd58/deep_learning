"""Task 1.1 end to end: ``python -m src.cli eda`` runs this and writes everything to outputs/eda/.

Covers the spec's EDA list: class distribution, image sizes and quality, caption lengths,
vocabulary and typos, missing data, duplicates and corrupted images, plus sample images,
augmentation examples and a full check of the reusable DataLoaders.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import audit, captions as cap
from src.data.hf import dataset_class_names, load_class_names, save_class_names
from src.data.split import load_split, manifest_path
from src.utils.config import output_dir


def run_eda(cfg: dict, data: dict, verify_loaders: bool = True, verbose: bool = True) -> dict:
    from src.visualization import augmentation, eda_plots

    out = output_dir(cfg, 'eda')
    review = output_dir(cfg, 'eda', 'review')
    train, test = data['train'], data['test']
    names = dataset_class_names(train, test)
    try:
        saved = load_class_names(cfg)
    except (FileNotFoundError, ValueError):
        saved = None
    if saved != names and len(names) == int(cfg['data']['num_classes']):
        save_class_names(cfg, names)  # keep configs/class_names.json identical to the dataset's own names
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    summary = {'images': {'train_original': len(train), 'test_official': len(test)}, 'species': len(names)}

    # 1. Class distribution and imbalance
    train_counts = pd.Series(train['label']).value_counts().sort_index()
    test_counts = pd.Series(test['label']).value_counts().sort_index()
    pd.DataFrame({'species': names, 'train': train_counts.values, 'test': test_counts.reindex(train_counts.index).values}
                 ).to_csv(out / 'class_counts.csv', index_label='label')
    summary['class_balance'] = {'train_min': int(train_counts.min()), 'train_max': int(train_counts.max()),
                                'train_mean': round(float(train_counts.mean()), 2),
                                'imbalance_ratio_max_over_min': round(float(train_counts.max() / train_counts.min()), 2),
                                'test_min': int(test_counts.min()), 'test_max': int(test_counts.max())}
    eda_plots.plot_class_distribution(train_counts, test_counts, out / '01_class_distribution.png')
    log('class distribution done')

    # 2. Image sizes, quality, corrupted files, duplicates
    meta = pd.concat([audit.scan_images(train, 'train'), audit.scan_images(test, 'test')], ignore_index=True)
    meta.drop(columns=['pixel_sha256']).to_csv(out / 'image_metadata.csv', index=False)
    flags = audit.quality_flags(meta)
    flags.drop(columns=['pixel_sha256']).to_csv(review / 'flagged_images.csv', index=False)
    eda_plots.plot_image_statistics(meta, flags, out / '02_image_sizes_and_quality.png')
    audit.contact_sheets(train, flags[flags.split == 'train'], review)
    corrupted = meta[~meta.decode_ok]
    labels = np.asarray(train['label'], dtype=int)
    train_ids, val_ids, split_hash = load_split(cfg, labels)
    split_of = {**{i: 'train' for i in train_ids}, **{i: 'validation' for i in val_ids}}
    duplicates = audit.duplicate_report(meta, split_of)
    (out / 'duplicates.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    ok = meta[meta.decode_ok]
    summary['image_quality'] = {
        'corrupted_files': int(len(corrupted)), 'flagged_for_review': int(len(flags)),
        'flag_reasons': flags['flag_reason'].str.split('; ').explode().value_counts().to_dict(),
        'width': ok['width'].describe().round(1).to_dict(), 'height': ok['height'].describe().round(1).to_dict(),
        'smaller_than_224': int((ok[['width', 'height']].min(axis=1) < 224).sum()),
        'non_rgb_modes': ok['mode'].value_counts().to_dict()}
    summary['duplicates'] = {k: duplicates[k] for k in ('exact_duplicate_groups', 'cross_split_groups')}
    summary['missing_data'] = [audit.missing_fields(train, 'train'), audit.missing_fields(test, 'test')]
    log('image audit done')

    # 3. Captions: cleaning, lengths, vocabulary, typos
    caption_results = {}
    for split, ds in (('train', train), ('test', test)):
        result = cap.analyse_captions(ds, split, out / 'captions')
        caption_results[split] = result
        summary[f'captions_{split}'] = result['summary']
    tr = caption_results['train']
    eda_plots.plot_caption_statistics(tr['captions'], tr['images'], out / '03_caption_lengths.png')
    eda_plots.plot_top_words(cap.top_words(tr['tokens'], 30), out / '04_top_words.png')
    typos = cap.typo_candidates(tr['tokens'])
    typos.to_csv(out / 'captions' / 'typo_candidates_train.csv', index=False)
    summary['captions_train']['typo_candidates'] = int(len(typos))
    summary['captions_train']['typo_examples'] = typos.head(10)[['word', 'likely_meant']].values.tolist()
    log('caption analysis done')

    # 4. Samples and augmentation examples
    eda_plots.plot_samples(train, names, out / '05_sample_images.png')
    rows = np.random.default_rng(1).choice(train_ids, size=4, replace=False)
    size = int(cfg['data']['image_size'])
    for recipe in ('transfer_mild', 'scratch_strong'):
        augmentation.plot_augmentations(train, rows, recipe, size, out / f'06_augmentation_{recipe}.png')
    log('samples and augmentation previews done')

    # 5. Split and DataLoader verification
    summary['split'] = {'manifest': str(manifest_path(cfg)), 'sha256': split_hash, 'train': len(train_ids),
                        'validation': len(val_ids), 'test': len(test),
                        'stratified_all_200_species_in_each': True}
    if verify_loaders:
        from src.data.loaders import build_loaders, verify_loader

        loaders = build_loaders(cfg, data, augmentation='scratch_strong')
        summary['loader_verification'] = {
            'image_size': loaders.image_size,
            'train': verify_loader(loaders.train, len(train_ids), loaders.image_size),
            'validation': verify_loader(loaders.validation, len(val_ids), loaders.image_size),
            'test': verify_loader(loaders.test, len(test), loaders.image_size)}
        log('full DataLoader verification done')

    (out / 'eda_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    log(f'EDA complete -> {out}')
    return summary
