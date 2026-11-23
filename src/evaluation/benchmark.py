"""Benchmark this project's models against published CUB-200-2011 results (spec section 5.11).

Our side comes from ``outputs/evaluation/*/metrics.json`` (run ``evaluate`` first); the published
side comes from ``configs/benchmark.yaml``. Both use the official CUB-200-2011 test split and
top-1 accuracy. Writes a table, a plot and ``benchmark_notes.md`` that explains fairness and
where the differences come from.
"""
from __future__ import annotations

import json

import pandas as pd
import yaml

from src.utils.config import output_dir, project_path


def build_benchmark(cfg: dict, benchmark_file='configs/benchmark.yaml', verbose=True) -> pd.DataFrame:
    spec = yaml.safe_load(project_path(benchmark_file).read_text(encoding='utf-8'))
    ours = []
    for path in sorted(output_dir(cfg, 'evaluation').glob('*/metrics.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        ours.append({'model': m['model'], 'accuracy': m['accuracy'], 'input_size': m['image_size'],
                     'backbone': m['architecture'], 'extra_annotations': False, 'source': 'this project',
                     'url': '', 'f1_macro': m.get('f1_macro')})
    if not ours:
        raise FileNotFoundError('No evaluated models found: run `python -m src.cli evaluate --model all` first')
    published = [{**{k: row.get(k) for k in ('model', 'accuracy', 'input_size', 'backbone',
                                              'extra_annotations', 'source', 'url')}, 'f1_macro': None}
                 for row in spec['published']]
    table = pd.DataFrame(ours + published).sort_values('accuracy', ascending=False).reset_index(drop=True)
    out = output_dir(cfg, 'benchmark')
    table.to_csv(out / 'benchmark.csv', index=False)
    from src.visualization.plots import plot_benchmark
    plot_benchmark(table, out / 'benchmark.png')
    (out / 'benchmark_notes.md').write_text(_notes(table, spec), encoding='utf-8')
    if verbose:
        print(table[['model', 'accuracy', 'input_size', 'source']].to_string(index=False))
        print(f'Saved to {out}')
    return table


def _notes(table: pd.DataFrame, spec: dict) -> str:
    ours = table[table.source == 'this project']
    best = ours.iloc[0]
    baseline = next((r for r in spec['published'] if 'baseline' in r['model'].lower()), None)
    lines = ['# Benchmark notes', '',
             '## How the comparison was made',
             '- **Same split:** every row uses the official CUB-200-2011 test set (5,794 images); our models were '
             'selected on our validation split only, never on the test set.',
             '- **Same metric:** top-1 accuracy, the number all the cited papers report. Our other metrics '
             '(macro F1, top-5) are in `outputs/comparison/`.',
             '- **Published numbers are cited, not re-run.** Each has a source and URL in `configs/benchmark.yaml`; '
             'check them against the papers before the report.', '',
             '## Results', '', '| Model | Top-1 accuracy | Input | Source |', '|---|---|---|---|']
    for row in table.itertuples():
        size = f'{int(row.input_size)}px' if pd.notna(row.input_size) else 'see paper'
        lines.append(f'| {row.model} | {row.accuracy * 100:.1f}% | {size} | {row.source} |')
    lines += ['', '## Why our numbers differ',
              f'- Our best model ({best.model}) reaches {best.accuracy * 100:.1f}%.' +
              (f' The plain fine-tuned ResNet-50 baseline in the literature reaches {baseline["accuracy"] * 100:.1f}%.'
               if baseline else ''),
              '- **Input resolution:** the published methods use 448x448 images; we use 224x224 (4x fewer pixels), '
              'which loses the small details that separate similar species.',
              '- **Model size:** ResNet-50 / ViT-B are much larger than our ResNet-18, EfficientNet-B0 and the '
              '5-block scratch CNN.',
              '- **Training budget:** our transfer models train for only 3 + 3 epochs and fine-tune just the last '
              'stage; published baselines fine-tune the whole network for many epochs.',
              '- **Specialised architectures:** NTS-Net and TransFG add part-localisation / patch-selection modules '
              'designed for fine-grained recognition; ViT/TransFG also start from the much larger ImageNet-21k.',
              '- **From scratch:** the scratch CNN has no pretraining at all, so it learns everything from 4,795 '
              'images, which is why it is far below every pretrained model.', '',
              '## Open-source models considered but not run']
    for item in spec.get('not_run', []):
        lines.append(f'- [{item["model"]}]({item["url"]}): {item["reason"]}')
    return '\n'.join(lines) + '\n'
