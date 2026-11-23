"""Command-line interface for the whole project (spec section 5.3). Run from the project root:

    python -m src.cli eda
    python -m src.cli tune     --config configs/tuning_scratch_cnn.yaml
    python -m src.cli train    --config configs/scratch_cnn.yaml --use-tuned
    python -m src.cli train    --config configs/resnet18.yaml
    python -m src.cli evaluate --model all
    python -m src.cli compare
    python -m src.cli benchmark
    python -m src.cli predict  --checkpoint checkpoints/resnet18_best.pt --image bird.jpg
    python -m src.cli gradcam  --checkpoint checkpoints/resnet18_best.pt --image bird.jpg

Every command accepts --task (only 'classification' exists in Milestone 1; Milestone 2 adds
'colorization' and 'caption') and --set section.key=value to override any config value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TASKS = ('classification',)
MODEL_CHOICES = ('all', 'scratch_cnn', 'resnet18', 'efficientnet_b0')


def _config(args, default='configs/data.yaml'):
    from src.utils.config import load_config

    return load_config(args.config or default, args.set)


def _tracker(cfg, args):
    from src.utils.tracking import make_tracker

    return make_tracker(cfg, enabled=not getattr(args, 'no_mlflow', False))


def cmd_check(args):
    """Offline sanity check: configs load, all three models build at 224px, present checkpoints load."""
    import torch

    from src.data.hf import load_class_names
    from src.models.registry import ARCHITECTURES, build_model, load_checkpoint
    from src.utils.config import checkpoint_dir, load_config
    from src.utils.seed import get_device

    cfg = _config(args)
    size = int(cfg['data']['image_size'])
    print(f'Python {sys.version.split()[0]} | torch {torch.__version__} | device {get_device()}')
    for name in ('scratch_cnn', 'resnet18', 'efficientnet_b0',
                 'tuning_scratch_cnn', 'tuning_resnet18', 'tuning_efficientnet_b0'):
        load_config(f'configs/{name}.yaml')
    print('configs: OK')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    for arch in ARCHITECTURES:
        model = build_model(arch, 200, pretrained=False).eval()
        with torch.inference_mode():
            shape = tuple(model(torch.zeros(1, 3, size, size)).shape)
        assert shape == (1, 200), shape
        print(f'{arch}: builds, {size}px input -> {shape}')
    names = load_class_names(cfg)
    for path in sorted(checkpoint_dir(cfg).glob('*.pt*')):
        loaded = load_checkpoint(path, fallback_class_names=names)
        print(f'checkpoint {path.name}: {loaded.display_name}, tag {loaded.tag}')
    print('check PASSED (no data downloaded, no training run)')


def cmd_eda(args):
    from src.data.eda import run_eda
    from src.data.hf import load_cub

    cfg = _config(args)
    run_eda(cfg, load_cub(cfg), verify_loaders=not args.skip_loader_check)


def cmd_augment_preview(args):
    import numpy as np

    from src.data.hf import load_cub
    from src.utils.config import output_dir
    from src.visualization.augmentation import plot_augmentations

    cfg = _config(args)
    train = load_cub(cfg, ('train',))['train']
    rows = np.random.default_rng(args.seed).choice(len(train), size=args.images, replace=False)
    for recipe in args.recipes:
        path = plot_augmentations(train, rows, recipe, int(cfg['data']['image_size']),
                                  output_dir(cfg, 'eda') / f'06_augmentation_{recipe}.png', seed=args.seed)
        print('Saved', path)


def cmd_train(args):
    from src.training.trainer import train_model
    from src.training.tuning import apply_tuned

    cfg = _config(args, None)
    if args.use_tuned:
        cfg = apply_tuned(cfg, args.tuned_file)
        print('Using tuned hyperparameters:', {k: cfg['training'][k] for k in ('lr', 'head_lr', 'weight_decay')
                                                if k in cfg['training']})
    summary = train_model(cfg, tracker=_tracker(cfg, args), run_name=args.run_name)
    print(json.dumps({k: v for k, v in summary.items() if k != 'history'}, indent=2, default=str))


def cmd_tune(args):
    from src.training.tuning import run_search

    cfg = _config(args, None)
    methods = ('grid', 'random') if args.method == 'both' else (args.method,)
    run_search(cfg, methods, tracker=_tracker(cfg, args))


def cmd_evaluate(args):
    from src.data.hf import load_cub
    from src.evaluation.evaluate import default_checkpoints, evaluate_checkpoint

    cfg = _config(args)
    paths = [Path(p) for p in args.checkpoint] if args.checkpoint else default_checkpoints(cfg, args.model)
    data = load_cub(cfg)
    results = [evaluate_checkpoint(cfg, path, data, gradcam_per_kind=args.gradcam_cases) for path in paths]
    print('\nSummary:')
    for r in results:
        print(f'  {r["model"]:28s} test accuracy {r["accuracy"]:.2%}  top-5 {r["top5_accuracy"]:.2%}  '
              f'macro F1 {r["f1_macro"]:.3f}')


def cmd_compare(args):
    from src.evaluation.evaluate import compare_models

    compare_models(_config(args), args.models)


def cmd_benchmark(args):
    from src.evaluation.benchmark import build_benchmark

    build_benchmark(_config(args), args.benchmark_file)


def _load_for_inference(args):
    from PIL import Image

    from src.data.hf import load_class_names
    from src.data.transforms import build_transform
    from src.models.registry import load_checkpoint
    from src.utils.seed import get_device

    cfg = _config(args)
    loaded = load_checkpoint(args.checkpoint, get_device(), fallback_class_names=load_class_names(cfg))
    image = Image.open(args.image)
    return cfg, loaded, build_transform(loaded.image_size, train=False)(image)


def cmd_predict(args):
    import torch

    if not 1 <= args.top_k <= 200:
        raise SystemExit('--top-k must be between 1 and 200')
    cfg, loaded, x = _load_for_inference(args)
    device = next(loaded.model.parameters()).device
    with torch.inference_mode():
        probs = loaded.model(x.unsqueeze(0).to(device)).softmax(1)[0]
    conf, ids = probs.topk(args.top_k)
    result = {'image': str(Path(args.image).resolve()), 'model': loaded.display_name,
              'predictions': [{'rank': r + 1, 'class_id': int(i), 'species': loaded.class_names[int(i)],
                               'probability': round(float(p), 4)} for r, (p, i) in enumerate(zip(conf, ids))]}
    print(f'{loaded.display_name} - top {args.top_k} for {Path(args.image).name}:')
    for p in result['predictions']:
        print(f'  {p["rank"]}. {p["species"]:32s} {p["probability"]:.1%}')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def cmd_gradcam(args):
    from src.utils.config import output_dir
    from src.visualization.gradcam import plot_single

    cfg, loaded, x = _load_for_inference(args)
    path = Path(args.output) if args.output else output_dir(cfg, 'gradcam') / f'{loaded.tag}_{Path(args.image).stem}.png'
    print('Saved Grad-CAM:', plot_single(loaded, x, path, args.class_id))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--task', choices=TASKS, default='classification',
                        help='Project task (Milestone 1: classification)')
    common.add_argument('--config', help='YAML config file (default: configs/data.yaml)')
    common.add_argument('--set', action='append', metavar='KEY=VALUE',
                        help='Override a config value, e.g. --set training.epochs=5 (repeatable)')

    parser = argparse.ArgumentParser(prog='python -m src.cli',
                                     description='CUB-200-2011 bird classification: EDA, training, tuning, '
                                                 'evaluation, Grad-CAM, comparison and benchmarking.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('check', parents=[common], help='Offline sanity check of configs, models and checkpoints')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('eda', parents=[common], help='Task 1.1: full EDA, caption cleaning, plots, loader checks')
    p.add_argument('--skip-loader-check', action='store_true', help='Skip iterating every DataLoader batch')
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser('augment-preview', parents=[common], help='Plot training augmentations for sample images')
    p.add_argument('--recipes', nargs='+', default=['transfer_mild', 'scratch_strong'])
    p.add_argument('--images', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.set_defaults(func=cmd_augment_preview)

    p = sub.add_parser('train', parents=[common], help='Train a model from its config (validation-based selection)')
    p.add_argument('--use-tuned', action='store_true', help='Apply outputs/tuning/<model>/best_params.yaml')
    p.add_argument('--tuned-file', help='Explicit best_params.yaml to apply')
    p.add_argument('--run-name', help='Name for the MLflow run and output folder')
    p.add_argument('--no-mlflow', action='store_true', help='Do not log this run to MLflow')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('tune', parents=[common], help='Grid and/or random hyperparameter search')
    p.add_argument('--method', choices=['grid', 'random', 'both'], default='both')
    p.add_argument('--no-mlflow', action='store_true')
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser('evaluate', parents=[common], help='Test-set metrics, confusion matrix and Grad-CAM')
    p.add_argument('--model', choices=MODEL_CHOICES, default='all', help='Which checkpoints/<model>_best.pt to use')
    p.add_argument('--checkpoint', nargs='+', help='Evaluate these checkpoint files instead of --model')
    p.add_argument('--gradcam-cases', type=int, default=4, help='Correct and wrong Grad-CAM examples (0 = none)')
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser('compare', parents=[common], help='Scratch vs transfer comparison of evaluated models')
    p.add_argument('--models', nargs='+', help='Evaluation tags to compare (default: all evaluated)')
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('benchmark', parents=[common], help='Compare against published CUB-200-2011 results')
    p.add_argument('--benchmark-file', default='configs/benchmark.yaml')
    p.set_defaults(func=cmd_benchmark)

    for name, func, text in (('predict', cmd_predict, 'Top-k species for one image'),
                             ('gradcam', cmd_gradcam, 'Grad-CAM heatmap for one image')):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument('--checkpoint', required=True, help='Any project checkpoint (.pt/.pth)')
        p.add_argument('--image', required=True, help='Path to a bird photo')
        p.add_argument('--output', help='Where to save the JSON (predict) or PNG (gradcam)')
        if name == 'predict':
            p.add_argument('--top-k', type=int, default=5)
        else:
            p.add_argument('--class-id', type=int, help='Class to explain (default: the predicted class)')
        p.set_defaults(func=func)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
