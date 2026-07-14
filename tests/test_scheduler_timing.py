"""Learning-rate timing contracts for native and MLX-reference Torch runs."""

from __future__ import annotations

import pytest
import torch

from src.train.schedulers import build_scheduler


def _optimizer(lr: float):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    return parameter, torch.optim.SGD([parameter], lr=lr)


def _update_lrs(optimizer, scheduler, parameter, updates: int):
    learning_rates = []
    for _ in range(updates):
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    return learning_rates


def _cfg(**overrides):
    cfg = {
        "lr": 1e-3,
        "min_lr": 1e-4,
        "epochs": 2,
        "warmup_epochs": 1,
        "cosine": True,
    }
    cfg.update(overrides)
    return cfg


def test_mlx_reference_torch_scheduler_matches_mlx_per_update_timing():
    pytest.importorskip("mlx.core")
    from src.runtime.backends import mlx_backend

    cfg = _cfg(reference_backend="mlx")
    parameter, optimizer = _optimizer(cfg["lr"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=4)
    mlx_scheduler = mlx_backend._build_scheduler(cfg, steps_per_epoch=4)

    torch_lrs = _update_lrs(optimizer, scheduler, parameter, updates=6)
    mlx_lrs = [mlx_scheduler.lr(step) for step in range(1, 7)]

    assert torch_lrs == pytest.approx(mlx_lrs)
    assert torch_lrs[0] == pytest.approx(cfg["lr"] / 4)


def test_native_torch_scheduler_keeps_historical_step_zero_start():
    cfg = _cfg()
    parameter, optimizer = _optimizer(cfg["lr"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=4)

    torch_lrs = _update_lrs(optimizer, scheduler, parameter, updates=2)

    assert torch_lrs == pytest.approx([0.0, cfg["lr"] / 4])
