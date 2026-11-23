"""Evaluate trained checkpoints on the official CUB-200 test split and compare them.

Outputs per model go to ``outputs/evaluation/<tag>/``; the scratch-vs-transfer comparison
goes to ``outputs/comparison/``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.hf import EXPECTED_SIZES, dataset_class_names, same_class_order
from src.data.loaders import build_loaders
from src.data.split import file_sha256, manifest_path
from src.evaluation.classification import align_predictions, metrics_for, most_confused_pairs, predict_loader
from src.models.registry import ARCHITECTURES, SCRATCH, load_checkpoint
from src.utils.config import checkpoint_dir, output_dir
from src.utils.seed import get_device


def _relative(path) -> str:
    """Path relative to the project root when possible (keeps personal folders out of committed results)."""
    from src.utils.config import PROJECT_ROOT

    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def default_checkpoints(cfg: dict, model: str = 'all') -> list[Path]:
    """checkpoints/<arch>_best.pt for each requested model; the legacy 128px scratch model too, if present."""
    folder = checkpoint_dir(cfg)
    archs = ARCHITECTURES if model == 'all' else (model,)
    found = []
    for arch in archs:
        if (folder / f'{arch}_best.pt').is_file():
            found.append(folder / f'{arch}_best.pt')
        if arch == SCRATCH and (folder / 'scratch_cnn_legacy128.pth').is_file():
            found.append(folder / 'scratch_cnn_legacy128.pth')
    if not found:
        raise FileNotFoundError(f'No checkpoints for {model!r} in {folder}. Download them (README) or train first.')
    return found


def evaluate_checkpoint(cfg: dict, path, data: dict, gradcam_per_kind: int = 4, verbose: bool = True) -> dict:
    test = data['test']
    names = dataset_class_names(data.get('train', test), test)
    device = get_device()
    loaded = load_checkpoint(path, device, fallback_class_names=names)
    if not same_class_order(loaded.class_names, names):
        raise ValueError(f'{path}: class order differs from the dataset; predictions would be mislabelled')
    current_split = file_sha256(manifest_path(cfg))
    trained_split = loaded.meta.get('split_sha256')
    if trained_split and trained_split != current_split:
        raise ValueError(f'{path} was trained on a different train/validation split')

    loaders = build_loaders(cfg, data, image_size=loaded.image_size, include=('test',))
    if verbose:
        print(f'Evaluating {loaded.display_name} [{loaded.tag}] on {len(loaders.test.dataset)} test images...', flush=True)
    predictions = predict_loader(loaded.model, loaders.test, device, test['file_name'])
    if len(predictions) != EXPECTED_SIZES['test'] or predictions.row_index.tolist() != list(range(EXPECTED_SIZES['test'])):
        raise ValueError('Predictions do not cover every official test image')

    metrics, cm, report = metrics_for(predictions, names)
    out = output_dir(cfg, 'evaluation', loaded.tag)
    output_dir(cfg, 'evaluation', 'predictions')
    predictions.to_csv(out.parent / 'predictions' / f'{loaded.tag}.csv', index=False)
    report.to_csv(out / 'per_class_report.csv')
    most_confused_pairs(cm, names).to_csv(out / 'most_confused_pairs.csv', index=False)
    import numpy as np
    np.save(out / 'confusion_matrix.npy', cm)
    from src.visualization.plots import plot_confusion
    plot_confusion(cm, out / 'confusion_matrix.png', f'{loaded.display_name}: test confusion matrix (row-normalised)')
    if gradcam_per_kind:
        from src.visualization.gradcam import plot_cases
        plot_cases(loaded, predictions, loaders.test.dataset, out / 'gradcam_correct_vs_wrong.png', gradcam_per_kind)

    record = {'tag': loaded.tag, 'model': loaded.display_name, 'architecture': loaded.architecture,
              'image_size': loaded.image_size, **metrics,
              'best_val_accuracy': loaded.meta.get('best_val_accuracy'),
              'checkpoint': _relative(path),
              'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'split_sha256': current_split,
              'split_hash_stored_in_checkpoint': bool(trained_split)}
    (out / 'metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if verbose:
        print(f'  accuracy {metrics["accuracy"]:.2%} | top-5 {metrics["top5_accuracy"]:.2%} | '
              f'macro F1 {metrics["f1_macro"]:.3f} -> {out}', flush=True)
    return record


def compare_models(cfg: dict, tags=None, verbose: bool = True) -> pd.DataFrame:
    """Scratch vs transfer table + plot, after checking all models saw identical test images."""
    eval_root = output_dir(cfg, 'evaluation')
    if not tags:
        tags = sorted(p.stem for p in (eval_root / 'predictions').glob('*.csv'))
    records, frames = [], []
    for tag in tags:
        pred_path, metric_path = eval_root / 'predictions' / f'{tag}.csv', eval_root / tag / 'metrics.json'
        if not pred_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f'{tag} has not been evaluated yet: run `python -m src.cli evaluate` first')
        frames.append((tag, pd.read_csv(pred_path)))
        records.append(json.loads(metric_path.read_text(encoding='utf-8')))
    align_predictions(frames)  # raises if models were evaluated on different images/labels
    columns = ['model', 'tag', 'image_size', 'accuracy', 'top5_accuracy', 'precision_macro', 'recall_macro',
               'f1_macro', 'f1_weighted', 'best_val_accuracy', 'samples']
    table = pd.DataFrame(records)[columns].sort_values('accuracy').reset_index(drop=True)
    out = output_dir(cfg, 'comparison')
    table.to_csv(out / 'model_comparison.csv', index=False)
    (out / 'model_comparison.md').write_text(_markdown_table(table), encoding='utf-8')
    from src.visualization.plots import plot_model_comparison
    plot_model_comparison(table, out / 'model_comparison.png')
    _per_class_gap(eval_root, table, out)
    if verbose:
        print('\nAll models were evaluated on the same official test images:\n' + table.to_string(index=False))
        print(f'Saved to {out}')
    return table


def _per_class_gap(eval_root: Path, table: pd.DataFrame, out: Path) -> None:
    """Species where the best model beats the weakest by the largest F1 margin (where pretraining helps most)."""
    if len(table) < 2:
        return
    weakest, best = table.iloc[0]['tag'], table.iloc[-1]['tag']
    read = lambda tag: pd.read_csv(eval_root / tag / 'per_class_report.csv', index_col=0)['f1-score']
    a, b = read(weakest), read(best)
    species = [i for i in a.index if i not in ('accuracy', 'macro avg', 'weighted avg')]
    gap = pd.DataFrame({f'f1_{weakest}': a[species], f'f1_{best}': b[species]})
    gap['difference'] = gap[f'f1_{best}'] - gap[f'f1_{weakest}']
    gap.sort_values('difference', ascending=False).to_csv(out / f'per_class_f1_{best}_vs_{weakest}.csv')


def _markdown_table(table: pd.DataFrame) -> str:
    pct = ['accuracy', 'top5_accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'best_val_accuracy']
    lines = ['| Model | Input | Test acc | Top-5 | Macro P | Macro R | Macro F1 | Weighted F1 | Val acc |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in table.itertuples():
        v = {c: (f'{getattr(row, c) * 100:.2f}%' if pd.notna(getattr(row, c)) else '-') for c in pct}
        lines.append(f'| {row.model} | {row.image_size}px | {v["accuracy"]} | {v["top5_accuracy"]} | '
                     f'{v["precision_macro"]} | {v["recall_macro"]} | {v["f1_macro"]} | {v["f1_weighted"]} | '
                     f'{v["best_val_accuracy"]} |')
    return '\n'.join(lines) + '\n'
