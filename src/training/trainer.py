"""Train any model from its config: data -> model -> stages -> best checkpoint, with MLflow tracking.

Model selection uses the validation split only; the official test split is never loaded here.
"""
from __future__ import annotations

import copy
import json
import time

import pandas as pd
import torch
from torch import nn

from src.data.hf import dataset_class_names, load_cub
from src.data.loaders import build_loaders
from src.models.registry import build_model, is_transfer, parameter_count, prepare_stage, save_checkpoint
from src.training.engine import run_epoch
from src.training.strategies import stages_for
from src.utils.config import checkpoint_dir, output_dir
from src.utils.seed import get_device, seed_everything
from src.utils.tracking import NullTracker, flatten


def _plain(obj):
    """Config copy containing only YAML/JSON-safe values (safe to store in a checkpoint)."""
    return json.loads(json.dumps(obj, default=str))


def train_model(cfg: dict, data: dict | None = None, tracker=None, save: bool = True,
                run_name: str | None = None, nested: bool = False, verbose: bool = True) -> dict:
    """Train the model described by ``cfg``; returns a summary (best scores, paths, history)."""
    tracker = tracker or NullTracker()
    seed_everything(int(cfg.get('seed', 42)))
    device = get_device()
    arch = cfg['model']['architecture']
    t = cfg['training']

    data = data or load_cub(cfg, ('train',))
    loaders = build_loaders(cfg, data, include=('train', 'validation'))
    class_names = dataset_class_names(data['train'])
    model = build_model(arch, num_classes=len(class_names), pretrained=bool(cfg['model'].get('pretrained', True)),
                        dropout=float(cfg['model'].get('dropout', 0.5))).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(t.get('label_smoothing', 0.0)))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    stages = stages_for(cfg)
    run_name = run_name or f'{arch}_{loaders.image_size}px'
    out = output_dir(cfg, 'training', run_name)

    if verbose:
        print(f'Device: {device} | model: {arch} | {parameter_count(model):,} parameters | '
              f'train {loaders.meta["train_images"]} / val {loaders.meta["val_images"]} images at '
              f'{loaders.image_size}px | augmentation: {loaders.meta["augmentation"]}', flush=True)

    rows, best = [], {'accuracy': -1.0, 'loss': float('inf'), 'state': None, 'epoch': 0, 'stage': None}
    started = time.perf_counter()
    with tracker.run(run_name, nested=nested, tags={'architecture': arch}):
        tracker.log_params({**flatten('training', t), **flatten('model', cfg['model']),
                            'data.image_size': loaders.image_size, 'data.augmentation': loaders.meta['augmentation'],
                            'data.batch_size': cfg['data']['batch_size'], 'data.train_images': loaders.meta['train_images'],
                            'data.val_images': loaders.meta['val_images'], 'split_sha256': loaders.split_sha256,
                            'seed': cfg.get('seed', 42), 'parameters_total': parameter_count(model)})
        global_epoch = 0
        for stage in stages:
            prepare_stage(model, arch, stage.name)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=stage.lr, weight_decay=stage.weight_decay)
            scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=stage.lr, epochs=stage.epochs,
                                                             steps_per_epoch=len(loaders.train))
                         if stage.schedule == 'onecycle' else None)
            if verbose:
                print(f'Stage {stage.name}: {stage.epochs} epochs, lr {stage.lr:g}, '
                      f'{parameter_count(model, True):,} trainable parameters', flush=True)
            for epoch in range(1, stage.epochs + 1):
                global_epoch += 1
                tick = time.perf_counter()
                tr = run_epoch(model, loaders.train, device, criterion, optimizer, scaler, scheduler,
                               t.get('grad_clip'), stage.freeze_bn)
                va = run_epoch(model, loaders.validation, device, criterion)
                row = {'stage': stage.name, 'epoch': global_epoch, 'train_loss': tr['loss'],
                       'train_accuracy': tr['accuracy'], 'val_loss': va['loss'], 'val_accuracy': va['accuracy'],
                       'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter() - tick}
                rows.append(row)
                tracker.log_metrics({k: row[k] for k in ('train_loss', 'train_accuracy', 'val_loss', 'val_accuracy')},
                                    step=global_epoch)
                improved = (va['accuracy'] > best['accuracy'] or
                            (va['accuracy'] == best['accuracy'] and va['loss'] < best['loss']))
                if improved:
                    best.update(accuracy=va['accuracy'], loss=va['loss'], epoch=global_epoch, stage=stage.name,
                                state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
                if verbose:
                    print(f'  epoch {global_epoch:3d} [{stage.name}] train loss {tr["loss"]:.4f} acc {tr["accuracy"]:.2%}'
                          f' | val loss {va["loss"]:.4f} acc {va["accuracy"]:.2%}'
                          f'{" *best*" if improved else ""} | {row["seconds"]:.0f}s', flush=True)
        if best['state'] is None:
            raise RuntimeError('Training produced no epochs; check the epoch settings')

        model.load_state_dict(best['state'])
        duration = time.perf_counter() - started
        history = pd.DataFrame(rows)
        history.to_csv(out / 'history.csv', index=False)
        from src.visualization.plots import plot_history
        curves = plot_history(history, out / 'learning_curves.png', f'{run_name}: train vs validation')
        summary = {'architecture': arch, 'run_name': run_name, 'image_size': loaders.image_size,
                   'best_val_accuracy': best['accuracy'], 'best_val_loss': best['loss'],
                   'best_epoch': best['epoch'], 'best_stage': best['stage'], 'epochs_run': global_epoch,
                   'duration_seconds': duration, 'split_sha256': loaders.split_sha256}
        tracker.log_metrics({'best_val_accuracy': best['accuracy'], 'best_val_loss': best['loss'],
                             'duration_seconds': duration})
        tracker.log_artifact(out / 'history.csv')
        tracker.log_artifact(curves)

        if save:
            path = save_checkpoint(checkpoint_dir(cfg) / t.get('checkpoint_name', f'{arch}_best.pt'), model, arch,
                                   class_names, loaders.image_size, epoch=best['epoch'], stage=best['stage'],
                                   best_val_accuracy=best['accuracy'], best_val_loss=best['loss'],
                                   split_sha256=loaders.split_sha256, augmentation=loaders.meta['augmentation'],
                                   config=_plain({k: cfg[k] for k in ('model', 'data', 'training') if k in cfg}))
            summary['checkpoint'] = str(path)
            tracker.log_params({'checkpoint_path': str(path)})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if verbose:
        print(f'Best validation accuracy {best["accuracy"]:.2%} at epoch {best["epoch"]} ({best["stage"]}); '
              f'outputs in {out}', flush=True)
    summary['history'] = history
    summary['is_transfer'] = is_transfer(arch)
    return summary
