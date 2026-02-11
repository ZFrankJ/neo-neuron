"""PyTorch backend adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ...models import LSTMLM, NeoLM, TransformerLM
from ...train import load_checkpoint, save_checkpoint
from ...train.eval import evaluate_metrics
from ...train.trainer import RestartEpoch, train_model
from ...utils import get_device, set_seed

NAME = "torch"


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Missing required config key: '{key}'")
    return cfg[key]


def build_model(cfg: Dict[str, Any], model_name: str):
    vocab_size = int(_require(cfg, "vocab_size"))
    d_model = int(_require(cfg, "d_model"))
    d_embed = int(cfg.get("d_embed", d_model))
    n_layers = int(cfg.get("n_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    tie_embeddings = bool(cfg.get("tie_embeddings", True))

    if model_name == "lstm":
        return LSTMLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            output_norm=str(cfg.get("output_norm", "layernorm")),
        )

    if model_name == "neo":
        cell_kwargs = {
            "output_norm": str(cfg.get("output_norm", "layernorm")),
            "activation_id": cfg.get("activation_id", "id3"),
        }
        return NeoLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            cell_type=str(cfg.get("cell_type", "cortical")),
            cell_kwargs=cell_kwargs,
            use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        )

    if model_name == "transformer":
        n_heads = int(_require(cfg, "n_heads"))
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=int(cfg.get("ff_mult", 4)),
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            max_seq_len=int(cfg.get("block_size", 2048)),
        )

    raise ValueError(f"Unknown model name '{model_name}'.")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_runtime_device(requested: Optional[str] = None):
    if requested and requested != "auto":
        return get_device(requested)
    return get_device()


def seed_all(seed: int) -> None:
    set_seed(int(seed))


def train_entry(
    model: torch.nn.Module,
    cfg: Any,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    test_ids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    return train_model(model, cfg, train_ids, val_ids, test_ids=test_ids, device=device)


def eval_metrics_entry(model: torch.nn.Module, ids: torch.Tensor, cfg: Any, device: torch.device):
    return evaluate_metrics(model, ids, cfg, device)


def load_checkpoint_entry(
    path: str | Path,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    device=None,
):
    return load_checkpoint(str(path), model, optimizer=optimizer, scheduler=scheduler, device=device)


def save_checkpoint_entry(
    path: str | Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    cfg: Any,
    best_val=None,
) -> None:
    save_checkpoint(
        str(path),
        model,
        optimizer,
        scheduler,
        epoch=epoch,
        global_step=global_step,
        cfg=cfg,
        best_val=best_val,
    )


def supports_probe() -> bool:
    return True
