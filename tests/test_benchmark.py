"""Benchmark table built from evaluation outputs plus configs/benchmark.yaml."""
import json

from src.evaluation.benchmark import build_benchmark
from src.utils.config import output_dir


def test_benchmark_combines_ours_and_published(tiny_cfg):
    folder = output_dir(tiny_cfg, 'evaluation', 'resnet18')
    (folder / 'metrics.json').write_text(json.dumps({'model': 'ResNet-18 (224px)', 'accuracy': 0.7,
                                                     'image_size': 224, 'architecture': 'resnet18',
                                                     'f1_macro': 0.69}))
    table = build_benchmark(tiny_cfg, verbose=False)
    assert 'this project' in set(table.source) and len(table) >= 2
    notes = (output_dir(tiny_cfg, 'benchmark') / 'benchmark_notes.md').read_text()
    assert 'Same split' in notes and 'Why our numbers differ' in notes
