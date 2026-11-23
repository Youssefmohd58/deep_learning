"""YAML configuration loading with `base:` inheritance and command-line overrides.

Every model config inherits shared data settings from ``configs/data.yaml`` through a
``base:`` key, so the dataset, split and image size are defined in exactly one place.
Relative paths are always resolved against the project root, so commands work no
matter which directory they are started from.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value) -> Path:
    """Resolve a path from a config relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` updated recursively with ``override`` (neither input is modified)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_bases(path: Path, seen: tuple = ()) -> dict:
    if path in seen:
        raise ValueError(f'Circular config inheritance: {path}')
    if not path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'{path} must contain a YAML mapping')
    base = raw.pop('base', None)
    if base is None:
        return raw
    return deep_merge(_read_with_bases((path.parent / base).resolve(), seen + (path,)), raw)


def parse_override(text: str) -> tuple[list[str], object]:
    """Parse ``section.key=value``; the value is read as YAML (numbers, bools, lists, null)."""
    if '=' not in text:
        raise ValueError(f'Override must look like key=value, got: {text}')
    key, value = text.split('=', 1)
    return key.strip().split('.'), yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or ():
        keys, value = parse_override(item)
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f'Cannot override inside non-mapping key in {item}')
        node[keys[-1]] = value
    return cfg


def load_config(path, overrides=None) -> dict:
    """Load a config file, resolve its ``base:`` chain and apply ``--set`` overrides."""
    path = project_path(path).resolve()
    cfg = apply_overrides(_read_with_bases(path), overrides)
    cfg['config_path'] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    data = cfg.get('data')
    if not isinstance(data, dict):
        raise ValueError("Config needs a 'data' section (inherit configs/data.yaml with base:)")
    for key in ('train_dataset', 'test_dataset', 'split_manifest', 'class_names_file',
                'num_classes', 'image_size', 'batch_size'):
        if key not in data:
            raise ValueError(f'Missing data.{key} in config')
    if int(data['num_classes']) != 200:
        raise ValueError('CUB-200-2011 has 200 classes; data.num_classes must be 200')
    if int(data['image_size']) < 32:
        raise ValueError('data.image_size is too small')


def output_dir(cfg: dict, *parts) -> Path:
    """Folder under the configured output root, created on demand."""
    root = project_path(cfg.get('project', {}).get('output_dir', 'outputs'))
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(cfg: dict) -> Path:
    path = project_path(cfg.get('project', {}).get('checkpoint_dir', 'checkpoints'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')
    return path
