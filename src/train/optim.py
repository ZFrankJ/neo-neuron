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
    alpha_lr = _cfg_get(cfg, "alpha_lr", None)
    alpha_lr_frac = _to_float(_cfg_get(cfg, "alpha_lr_frac", 0.1), 0.1)
    alpha_lr = _to_float(alpha_lr, lr * alpha_lr_frac) if alpha_lr is not None else lr * alpha_lr_frac

    alpha_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "alpha" in name:
            alpha_params.append(param)
        else:
            other_params.append(param)

    if alpha_params:
        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                {"params": alpha_params, "lr": alpha_lr, "weight_decay": 0.0},
            ]
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
