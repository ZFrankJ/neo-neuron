"""Learning-rate schedule helpers."""

import math
from typing import Any, Optional

import torch


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    steps_per_epoch: int,
    total_epochs: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if not _cfg_get(cfg, "cosine", True):
        return None

    epochs = total_epochs if total_epochs is not None else _cfg_get(cfg, "epochs", 1)
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_epochs = _cfg_get(cfg, "warmup_epochs", 0)
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    base_lr = _cfg_get(cfg, "lr", 3e-4)
    min_lr = _cfg_get(cfg, "min_lr", 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
