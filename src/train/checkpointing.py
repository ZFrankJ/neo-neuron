"""Checkpoint save/load helpers."""

import pickle
from pathlib import Path
from typing import Any, Optional

import torch

from ..runtime.checkpoint_compat import (
    infer_checkpoint_backend,
    infer_model_name_from_model,
    load_checkpoint_payload,
    map_model_state,
    to_numpy_state_dict,
    torch_template,
)


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
        "format": "neo_unified_checkpoint_v1",
        "backend": "torch",
        "model_state_dict": to_numpy_state_dict(model.state_dict()),
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
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> dict:
    ckpt = load_checkpoint_payload(path, map_location=device)
    raw_state = ckpt.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint missing 'model_state_dict' mapping.")

    src_backend = infer_checkpoint_backend(ckpt)
    model_name = infer_model_name_from_model(model)
    dst_template, dtypes = torch_template(model)
    mapped_state, _ = map_model_state(
        model_name=model_name,
        src_backend=src_backend,
        dst_backend="torch",
        src_state_np=to_numpy_state_dict(raw_state),
        dst_template=dst_template,
        cfg=ckpt.get("cfg", {}),
    )
    torch_state = {}
    for k, dv in dst_template.items():
        t = torch.from_numpy(mapped_state[k]).to(dtype=dtypes[k])
        if device is not None:
            t = t.to(device=device)
        if tuple(t.shape) != tuple(dv.shape):
            raise ValueError(f"Converted shape mismatch for '{k}': {tuple(t.shape)} vs {tuple(dv.shape)}")
        torch_state[k] = t
    model.load_state_dict(torch_state)

    # Optimizer/scheduler restoration remains backend-native only.
    if src_backend == "torch":
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
