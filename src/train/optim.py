"""Optimizer helpers."""

from typing import Any

import torch


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    lr = _to_float(_cfg_get(cfg, "lr", 3e-4), 3e-4)
    weight_decay = _to_float(_cfg_get(cfg, "weight_decay", 0.0), 0.0)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
