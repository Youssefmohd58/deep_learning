"""Experiment tracking with MLflow (spec section 5.7).

All runs go to one local MLflow store under ``outputs/`` so every member can open them with::

    mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db

``NullTracker`` has the same interface and is used by tests and by ``--no-mlflow``.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from src.utils.config import output_dir


class NullTracker:
    """Does nothing; lets training code call tracking methods unconditionally."""

    @contextmanager
    def run(self, name, nested=False, tags=None):
        yield self

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path):
        pass

    def log_dict(self, obj, name):
        pass


class MLflowTracker(NullTracker):
    def __init__(self, cfg: dict):
        import mlflow

        root = output_dir(cfg)
        self.mlflow = mlflow
        mlflow.set_tracking_uri(f'sqlite:///{(root / "mlflow.db").resolve().as_posix()}')
        name = cfg.get('project', {}).get('experiment', 'cub200_milestone1')
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(root / 'mlartifacts').resolve().as_uri())
        mlflow.set_experiment(name)

    @contextmanager
    def run(self, name, nested=False, tags=None):
        with self.mlflow.start_run(run_name=name, nested=nested, tags=tags or {}):
            yield self

    def log_params(self, params):
        clean = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
        self.mlflow.log_params(clean)

    def log_metrics(self, metrics, step=None):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path):
        if Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def log_dict(self, obj, name):
        self.mlflow.log_dict(obj, name)


def make_tracker(cfg: dict, enabled: bool = True):
    if not enabled or not cfg.get('tracking', {}).get('enabled', True):
        return NullTracker()
    return MLflowTracker(cfg)


def flatten(prefix: str, obj) -> dict:
    """Flatten nested config sections into MLflow-friendly ``a.b.c`` parameter names."""
    if not isinstance(obj, dict):
        return {prefix: obj}
    out = {}
    for key, value in obj.items():
        out.update(flatten(f'{prefix}.{key}' if prefix else str(key), value))
    return out
