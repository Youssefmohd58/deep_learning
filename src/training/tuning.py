"""Hyperparameter tuning with Grid Search and Random Search (spec section 5.8).

Both methods search the same hyperparameters for the same model, are given the same
per-trial budget (epochs, data) and are scored on the validation split only. The run
writes every trial, a method comparison (best score, number of trials, total time),
a plot, and ``best_params.yaml`` that ``train --use-tuned`` applies.
"""
from __future__ import annotations

import itertools
import math
import random

import pandas as pd

from src.data.hf import load_cub
from src.training.trainer import train_model
from src.utils.config import apply_overrides, output_dir, project_path, save_yaml
from src.utils.tracking import NullTracker


def grid_trials(grid: dict) -> list[dict]:
    """Every combination of the listed values."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _sample(spec, rng: random.Random):
    if isinstance(spec, list):
        return rng.choice(spec)
    if 'log_uniform' in spec:
        low, high = spec['log_uniform']
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    if 'uniform' in spec:
        return rng.uniform(*spec['uniform'])
    if 'choice' in spec:
        return rng.choice(spec['choice'])
    raise ValueError(f'Unknown search-space spec {spec!r}; use log_uniform, uniform, choice or a list')


def random_trials(space: dict, trials: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [{key: _sample(spec, rng) for key, spec in space.items()} for _ in range(int(trials))]


def trial_config(cfg: dict, params: dict) -> dict:
    """Config for one trial: the searched values plus the shorter tuning budget."""
    tuning = cfg['tuning']
    overrides = [f'training.{k}={v}' for k, v in params.items()]
    if cfg['training']['strategy'] == 'scratch':
        overrides.append(f'training.epochs={tuning["epochs"]}')
    else:  # transfer: tune the head stage, skip fine-tuning to keep trials comparable and cheap
        overrides += [f'training.head_epochs={tuning["epochs"]}', 'training.finetune_epochs=0']
    for key in ('max_train_images', 'max_val_images'):
        if tuning.get(key) is not None:
            overrides.append(f'data.{key}={tuning[key]}')
    return apply_overrides(cfg, overrides)


def _python_value(value):
    """NumPy scalar -> plain Python number so it can be written to YAML."""
    return value.item() if hasattr(value, 'item') else value


def best_params_path(cfg: dict):
    return output_dir(cfg, 'tuning', cfg['model']['architecture']) / 'best_params.yaml'


def run_search(cfg: dict, methods=('grid', 'random'), data=None, tracker=None, verbose=True) -> dict:
    if 'tuning' not in cfg:
        raise ValueError('This config has no tuning section; use configs/tuning_<model>.yaml')
    tracker = tracker or NullTracker()
    arch = cfg['model']['architecture']
    data = data or load_cub(cfg, ('train',))
    plans = {'grid': grid_trials(cfg['tuning']['grid']),
             'random': random_trials(cfg['tuning']['random']['space'], cfg['tuning']['random']['trials'],
                                     int(cfg.get('seed', 42)))}
    out = output_dir(cfg, 'tuning', arch)
    rows = []
    with tracker.run(f'tuning_{arch}', tags={'architecture': arch, 'kind': 'tuning'}):
        tracker.log_params({'methods': list(methods), 'trial_epochs': cfg['tuning']['epochs'],
                            'grid': cfg['tuning']['grid'], 'random_space': cfg['tuning']['random']['space']})
        for method in methods:
            for trial, params in enumerate(plans[method], 1):
                if verbose:
                    shown = ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in params.items())
                    print(f'\n[{method} search] trial {trial}/{len(plans[method])}: {shown}', flush=True)
                summary = train_model(trial_config(cfg, params), data=data, tracker=tracker, save=False,
                                      run_name=f'tune_{arch}_{method}_{trial:02d}', nested=True,
                                      verbose=verbose, output_group='tuning_runs')
                rows.append({'method': method, 'trial': trial, **params,
                             'best_val_accuracy': summary['best_val_accuracy'],
                             'best_val_loss': summary['best_val_loss'], 'best_epoch': summary['best_epoch'],
                             'duration_seconds': summary['duration_seconds']})
                pd.DataFrame(rows).to_csv(out / 'trials.csv', index=False)  # save progress after each trial

        trials = pd.DataFrame(rows)
        comparison = (trials.groupby('method', sort=False)
                      .agg(best_val_accuracy=('best_val_accuracy', 'max'),
                           mean_val_accuracy=('best_val_accuracy', 'mean'),
                           trials=('trial', 'count'), total_seconds=('duration_seconds', 'sum'))
                      .reset_index())
        comparison.to_csv(out / 'grid_vs_random.csv', index=False)
        best_row = trials.sort_values(['best_val_accuracy', 'best_val_loss'], ascending=[False, True]).iloc[0]
        searched = list(cfg['tuning']['grid']) + [k for k in cfg['tuning']['random']['space']
                                                  if k not in cfg['tuning']['grid']]
        best = {k: _python_value(best_row[k]) for k in searched}
        save_yaml(out / 'best_params.yaml', {'architecture': arch, 'found_by': str(best_row['method']),
                                             'best_val_accuracy': float(best_row['best_val_accuracy']),
                                             'training': best})
        from src.visualization.plots import plot_tuning
        plot = plot_tuning(trials, out / 'grid_vs_random.png')
        for name in ('trials.csv', 'grid_vs_random.csv', 'best_params.yaml'):
            tracker.log_artifact(out / name)
        tracker.log_artifact(plot)
        tracker.log_metrics({f'{r.method}_best_val_accuracy': r.best_val_accuracy for r in comparison.itertuples()})
    if verbose:
        print('\nGrid vs random search:\n' + comparison.to_string(index=False))
        print(f'Best: {best} ({best_row["method"]}, val acc {best_row["best_val_accuracy"]:.2%}) -> {out / "best_params.yaml"}')
    return {'trials': trials, 'comparison': comparison, 'best': best, 'output_dir': out}


def apply_tuned(cfg: dict, path=None) -> dict:
    """Return ``cfg`` with the tuned training values from best_params.yaml applied."""
    import yaml

    path = project_path(path) if path else best_params_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f'No tuning results at {path}. Run `python -m src.cli tune` first.')
    tuned = yaml.safe_load(path.read_text(encoding='utf-8'))
    if tuned.get('architecture') != cfg['model']['architecture']:
        raise ValueError(f'{path} was tuned for {tuned.get("architecture")}, not {cfg["model"]["architecture"]}')
    return apply_overrides(cfg, [f'training.{k}={v}' for k, v in tuned['training'].items()])
