"""Cross-backend checkpoint compatibility helpers."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def load_checkpoint_payload(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    p = Path(path)
    try:
        obj = torch.load(str(p), map_location=map_location, weights_only=False)
        if isinstance(obj, dict):
            return obj
        return {"model_state_dict": obj}
    except Exception:
        with p.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Unsupported checkpoint payload type.")


def infer_checkpoint_backend(payload: Dict[str, Any]) -> str:
    if payload.get("backend") == "mlx":
        return "mlx"
    state = payload.get("model_state_dict")
    if isinstance(state, dict):
        for v in state.values():
            if isinstance(v, torch.Tensor):
                return "torch"
    return "torch"


def infer_model_name_from_model(model: Any) -> str:
    if hasattr(model, "recurrent"):
        return "neo"
    if hasattr(model, "lstm") or hasattr(model, "lstm_layers"):
        return "lstm"
    if hasattr(model, "blocks") or hasattr(model, "encoder"):
        return "transformer"
    raise ValueError("Unable to infer model family from model object.")


def to_numpy_state_dict(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


def torch_template(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.dtype]]:
    sd = model.state_dict()
    dtypes = {k: v.dtype for k, v in sd.items()}
    return sd, dtypes


def _identity_map(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, dv in dst_template.items():
        if k not in src:
            raise KeyError(f"Missing key '{k}' in source state dict.")
        sv = src[k]
        if tuple(sv.shape) != tuple(dv.shape):
            raise ValueError(f"Shape mismatch for '{k}': src {tuple(sv.shape)} vs dst {tuple(dv.shape)}")
        out[k] = sv
    for k in sorted(set(src.keys()) - set(dst_template.keys())):
        warn.append(f"Ignoring extra source key: {k}")
    return out


def _map_lstm_torch_to_mlx(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    common = [
        "emb.weight",
        "in_proj.weight",
        "in_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
        "output_bias",
        "head.weight",
        "head.bias",
        "out_norm.weight",
        "out_norm.bias",
    ]
    for k in common:
        if k in src and k in dst_template:
            out[k] = src[k]

    for i in range(n_layers):
        out[f"lstm_layers.{i}.Wx"] = src[f"lstm.weight_ih_l{i}"]
        out[f"lstm_layers.{i}.Wh"] = src[f"lstm.weight_hh_l{i}"]
        out[f"lstm_layers.{i}.bias"] = src[f"lstm.bias_ih_l{i}"] + src[f"lstm.bias_hh_l{i}"]

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    for k in sorted(set(src.keys()) - set(out.keys())):
        warn.append(f"Ignoring extra source key: {k}")
    return out


def _map_lstm_mlx_to_torch(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    common = [
        "emb.weight",
        "in_proj.weight",
        "in_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
        "output_bias",
        "head.weight",
        "head.bias",
        "out_norm.weight",
        "out_norm.bias",
    ]
    for k in common:
        if k in src and k in dst_template:
            out[k] = src[k]

    for i in range(n_layers):
        wx = src[f"lstm_layers.{i}.Wx"]
        wh = src[f"lstm_layers.{i}.Wh"]
        b = src[f"lstm_layers.{i}.bias"]
        out[f"lstm.weight_ih_l{i}"] = wx
        out[f"lstm.weight_hh_l{i}"] = wh
        out[f"lstm.bias_ih_l{i}"] = b
        out[f"lstm.bias_hh_l{i}"] = np.zeros_like(b)

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    for k in sorted(set(src.keys()) - set(out.keys())):
        warn.append(f"Ignoring extra source key: {k}")
    return out


def _map_transformer_torch_to_mlx(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    d_model: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    pairs = [
        ("emb.weight", "emb.weight"),
        ("pos_emb.weight", "pos_emb.weight"),
        ("ln_f.weight", "encoder.ln.weight"),
        ("ln_f.bias", "encoder.ln.bias"),
        ("output_bias", "output_bias"),
    ]
    for s, d in pairs:
        if s in src and d in dst_template:
            out[d] = src[s]

    for i in range(n_layers):
        qkv_w = src[f"blocks.{i}.attn.qkv.weight"]
        out[f"encoder.layers.{i}.attention.query_proj.weight"] = qkv_w[0:d_model, :]
        out[f"encoder.layers.{i}.attention.key_proj.weight"] = qkv_w[d_model : 2 * d_model, :]
        out[f"encoder.layers.{i}.attention.value_proj.weight"] = qkv_w[2 * d_model : 3 * d_model, :]
        out[f"encoder.layers.{i}.attention.out_proj.weight"] = src[f"blocks.{i}.attn.out_proj.weight"]
        out[f"encoder.layers.{i}.ln1.weight"] = src[f"blocks.{i}.ln1.weight"]
        out[f"encoder.layers.{i}.ln1.bias"] = src[f"blocks.{i}.ln1.bias"]
        out[f"encoder.layers.{i}.ln2.weight"] = src[f"blocks.{i}.ln2.weight"]
        out[f"encoder.layers.{i}.ln2.bias"] = src[f"blocks.{i}.ln2.bias"]
        out[f"encoder.layers.{i}.linear1.weight"] = src[f"blocks.{i}.mlp.0.weight"]
        out[f"encoder.layers.{i}.linear1.bias"] = src[f"blocks.{i}.mlp.0.bias"]
        out[f"encoder.layers.{i}.linear2.weight"] = src[f"blocks.{i}.mlp.3.weight"]
        out[f"encoder.layers.{i}.linear2.bias"] = src[f"blocks.{i}.mlp.3.bias"]
        warn.append(f"Dropping torch attention biases for layer {i} (MLX transformer has no attention bias params).")

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def _map_transformer_mlx_to_torch(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    d_model: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    pairs = [
        ("emb.weight", "emb.weight"),
        ("pos_emb.weight", "pos_emb.weight"),
        ("encoder.ln.weight", "ln_f.weight"),
        ("encoder.ln.bias", "ln_f.bias"),
        ("output_bias", "output_bias"),
    ]
    for s, d in pairs:
        if s in src and d in dst_template:
            out[d] = src[s]

    for i in range(n_layers):
        qw = src[f"encoder.layers.{i}.attention.query_proj.weight"]
        kw = src[f"encoder.layers.{i}.attention.key_proj.weight"]
        vw = src[f"encoder.layers.{i}.attention.value_proj.weight"]
        out[f"blocks.{i}.attn.qkv.weight"] = np.concatenate([qw, kw, vw], axis=0)
        out[f"blocks.{i}.attn.qkv.bias"] = np.zeros((3 * d_model,), dtype=qw.dtype)
        out[f"blocks.{i}.attn.out_proj.weight"] = src[f"encoder.layers.{i}.attention.out_proj.weight"]
        out[f"blocks.{i}.attn.out_proj.bias"] = np.zeros((d_model,), dtype=qw.dtype)
        out[f"blocks.{i}.ln1.weight"] = src[f"encoder.layers.{i}.ln1.weight"]
        out[f"blocks.{i}.ln1.bias"] = src[f"encoder.layers.{i}.ln1.bias"]
        out[f"blocks.{i}.ln2.weight"] = src[f"encoder.layers.{i}.ln2.weight"]
        out[f"blocks.{i}.ln2.bias"] = src[f"encoder.layers.{i}.ln2.bias"]
        out[f"blocks.{i}.mlp.0.weight"] = src[f"encoder.layers.{i}.linear1.weight"]
        out[f"blocks.{i}.mlp.0.bias"] = src[f"encoder.layers.{i}.linear1.bias"]
        out[f"blocks.{i}.mlp.3.weight"] = src[f"encoder.layers.{i}.linear2.weight"]
        out[f"blocks.{i}.mlp.3.bias"] = src[f"encoder.layers.{i}.linear2.bias"]
        warn.append(f"Filling torch attention biases with zeros for layer {i} (MLX source has no attention bias params).")

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def map_model_state(
    model_name: str,
    src_backend: str,
    dst_backend: str,
    src_state_np: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    cfg = cfg or {}
    warn: List[str] = []

    if src_backend == dst_backend:
        mapped = _identity_map(src_state_np, dst_template, warn)
    elif model_name == "neo":
        mapped = _identity_map(src_state_np, dst_template, warn)
    elif model_name == "lstm":
        n_layers = int(cfg.get("n_layers", 1))
        if src_backend == "torch" and dst_backend == "mlx":
            mapped = _map_lstm_torch_to_mlx(src_state_np, dst_template, n_layers, warn)
        elif src_backend == "mlx" and dst_backend == "torch":
            mapped = _map_lstm_mlx_to_torch(src_state_np, dst_template, n_layers, warn)
        else:
            raise ValueError(f"Unsupported LSTM mapping {src_backend} -> {dst_backend}")
    elif model_name == "transformer":
        n_layers = int(cfg.get("n_layers", 1))
        d_model = int(cfg["d_model"])
        if src_backend == "torch" and dst_backend == "mlx":
            mapped = _map_transformer_torch_to_mlx(src_state_np, dst_template, n_layers, d_model, warn)
        elif src_backend == "mlx" and dst_backend == "torch":
            mapped = _map_transformer_mlx_to_torch(src_state_np, dst_template, n_layers, d_model, warn)
        else:
            raise ValueError(f"Unsupported Transformer mapping {src_backend} -> {dst_backend}")
    else:
        raise ValueError(f"Unknown model name '{model_name}'")

    for msg in warn:
        warnings.warn(msg)
    return mapped, warn

