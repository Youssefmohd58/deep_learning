"""Training strategies: how a model is trained, expressed as a list of stages.

* scratch  - one stage; every layer trains from random initialisation with AdamW and a
             OneCycle learning-rate schedule (warm-up then cosine decay), as Member 2 designed.
* transfer - Member 3's two stages: (1) 'head': only the new 200-class classifier trains on
             top of frozen ImageNet features; (2) 'finetune': the last convolutional stage is
             unfrozen too and trained with a 10x smaller learning rate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    epochs: int
    lr: float
    weight_decay: float
    schedule: str | None = None   # 'onecycle' or None (constant learning rate)
    freeze_bn: bool = False


def stages_for(cfg: dict) -> list[Stage]:
    t = cfg['training']
    strategy = t.get('strategy')
    if strategy == 'scratch':
        return [Stage('full', int(t['epochs']), float(t['lr']), float(t['weight_decay']), 'onecycle')]
    if strategy == 'transfer':
        stages = [Stage('head', int(t['head_epochs']), float(t['head_lr']), float(t['weight_decay']), None, True),
                  Stage('finetune', int(t['finetune_epochs']), float(t['finetune_lr']), float(t['weight_decay']),
                        None, True)]
        return [s for s in stages if s.epochs > 0]
    raise ValueError(f"training.strategy must be 'scratch' or 'transfer', got {strategy!r}")
