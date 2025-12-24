"""Checkpoint save/load helpers."""

from typing import Any, Optional

import torch


def _cfg_to_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {key: getattr(cfg, key) for key in dir(cfg) if not key.startswith("_")}


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    cfg: Any,
    best_val: Optional[float] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "cfg": _cfg_to_dict(cfg),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if best_val is not None:
        payload["best_val"] = best_val
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
