"""Optimizer helpers."""

from typing import Any, Dict, List

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
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


def _to_betas(value: Any, default=(0.9, 0.95)) -> tuple:
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
    except (TypeError, ValueError):
        return default
    return default


def _get_module_map(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    return dict(model.named_modules())


def _is_norm_module(module: torch.nn.Module) -> bool:
    rms_norm = getattr(nn, "RMSNorm", None)
    norm_types = (nn.LayerNorm,)
    if rms_norm is not None:
        norm_types = (nn.LayerNorm, rms_norm)
    return isinstance(module, norm_types)


def _build_weight_decay_groups(
    model: torch.nn.Module,
    cfg: Any,
    lr: float,
) -> List[Dict[str, object]]:
    module_map = _get_module_map(model)

    embed_wd = _to_float(_cfg_get(cfg, "embed_weight_decay", 0.0), 0.0)
    proj_wd = _to_float(_cfg_get(cfg, "proj_weight_decay", 1e-3), 1e-3)
    recurrent_wd = _to_float(_cfg_get(cfg, "recurrent_weight_decay", 0.0), 0.0)
    transformer_wd = _to_float(_cfg_get(cfg, "transformer_weight_decay", 1e-2), 1e-2)
    no_decay: List[torch.nn.Parameter] = []
    embeddings: List[torch.nn.Parameter] = []
    recurrent: List[torch.nn.Parameter] = []
    proj: List[torch.nn.Parameter] = []

    is_transformer = hasattr(model, "blocks")
    is_lstm = hasattr(model, "lstm")
    is_neo = hasattr(model, "recurrent")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "output_bias" or name.endswith(".output_bias"):
            no_decay.append(param)
            continue
        leaf = name.split(".")[-1]
        if name.endswith(".bias") or leaf.startswith("bias_"):
            no_decay.append(param)
            continue
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = module_map.get(module_name)
        if module is not None:
            if _is_norm_module(module):
                no_decay.append(param)
                continue
            if isinstance(module, nn.Embedding):
                embeddings.append(param)
                continue

        if is_lstm and name.startswith("lstm."):
            recurrent.append(param)
        elif is_neo and name.startswith("recurrent."):
            recurrent.append(param)
        else:
            proj.append(param)

    param_groups: List[Dict[str, object]] = []

    if is_transformer:
        decay = proj
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": transformer_wd})
    else:
        if proj:
            param_groups.append({"params": proj, "lr": lr, "weight_decay": proj_wd})
        if recurrent:
            param_groups.append({"params": recurrent, "lr": lr, "weight_decay": recurrent_wd})

    if embeddings:
        param_groups.append({"params": embeddings, "lr": lr, "weight_decay": embed_wd})
    if no_decay:
        param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return param_groups


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    lr = _to_float(_cfg_get(cfg, "lr", 3e-4), 3e-4)
    weight_decay = _to_float(_cfg_get(cfg, "weight_decay", 0.0), 0.0)
    betas = _to_betas(_cfg_get(cfg, "betas", (0.9, 0.95)), (0.9, 0.95))
    adam_eps = _to_float(_cfg_get(cfg, "adam_eps", 1e-8), 1e-8)

    policy = str(_cfg_get(cfg, "weight_decay_policy", "table")).lower()
    if policy in ("table", "per_param", "per-parameter"):
        param_groups = _build_weight_decay_groups(model, cfg, lr)
        return torch.optim.AdamW(param_groups, betas=betas, eps=adam_eps)

    module_map = _get_module_map(model)
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "output_bias" or name.endswith(".output_bias"):
            no_decay.append(param)
            continue
        leaf = name.split(".")[-1]
        if name.endswith(".bias") or leaf.startswith("bias_"):
            no_decay.append(param)
            continue
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = module_map.get(module_name)
        if module is not None:
            if _is_norm_module(module) or isinstance(module, nn.Embedding):
                no_decay.append(param)
                continue
        decay.append(param)

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, betas=betas, eps=adam_eps)
