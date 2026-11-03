"""Configuration, reproducibility and output paths for Member 3 CLI."""
from pathlib import Path
import json
import random
import numpy as np
import torch
import yaml


def load_config(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Configuration file not found: {path}')
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):
        raise ValueError('YAML config must contain a mapping')
    required = ('seed','image_size','batch_size','num_workers','num_classes',
                'validation_fraction','output_dir','head_epochs','finetune_epochs',
                'head_lr','finetune_lr','weight_decay','label_smoothing',
                'tuning_epochs','tuning_train_images','tuning_val_images')
    missing=set(required)-set(cfg)
    if missing:
        raise ValueError(f'Missing config keys: {sorted(missing)}')
    root = path.parent.parent if path.parent.name=='configs' else path.parent
    cfg['project_root'] = str(root)
    output = Path(cfg['output_dir'])
    cfg['output_dir'] = str(output if output.is_absolute() else root / output)
    for key in ('split_manifest','scratch_checkpoint','class_names_file'):
        if cfg.get(key):
            value=Path(cfg[key])
            cfg[key]=str(value if value.is_absolute() else root/value)
    if int(cfg['num_classes']) != 200 or not 0 < float(cfg['validation_fraction']) < 1:
        raise ValueError('Expected 200 classes and a validation_fraction between 0 and 1')
    return cfg


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def prepare_output(cfg):
    root=Path(cfg['output_dir'])
    for name in ('data','checkpoints','figures','predictions','metrics','history','tracking'):
        (root/name).mkdir(parents=True,exist_ok=True)
    (root/'run_config.json').write_text(json.dumps(cfg,indent=2,default=str),encoding='utf-8')
    return root
