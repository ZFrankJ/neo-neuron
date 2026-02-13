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


def _infer_n_layers_from_template_keys(dst_template: Dict[str, Any]) -> int:
    max_idx = -1
    for k in dst_template.keys():
        if not k.startswith("recurrent.layers."):
            continue
        parts = k.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def _map_neo_state(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    cfg: Dict[str, Any],
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # First copy all keys that already match by name and shape.
    for k, dv in dst_template.items():
        sv = src.get(k)
        if sv is not None and tuple(sv.shape) == tuple(dv.shape):
            out[k] = sv

    n_layers = int(cfg.get("n_layers", 0)) or _infer_n_layers_from_template_keys(dst_template)

    # Backward-compat for old per-cell norm naming:
    # recurrent.layers.{i}.out_norm.{weight,bias} -> recurrent.pre_norms.{i}.{weight,bias}
    for i in range(n_layers):
        for suffix in ("weight", "bias"):
            dst_key = f"recurrent.pre_norms.{i}.{suffix}"
            if dst_key in out or dst_key not in dst_template:
                continue
            src_key = f"recurrent.layers.{i}.out_norm.{suffix}"
            if src_key in src and tuple(src[src_key].shape) == tuple(dst_template[dst_key].shape):
                out[dst_key] = src[src_key]

    # Backward-compat for final stack norm:
    # recurrent.layers.{last}.out_norm.{weight,bias} -> recurrent.stack_norm.{weight,bias}
    for suffix in ("weight", "bias"):
        dst_key = f"recurrent.stack_norm.{suffix}"
        if dst_key in out or dst_key not in dst_template:
            continue
        src_key_direct = dst_key
        src_key_fallback = f"recurrent.layers.{max(0, n_layers - 1)}.out_norm.{suffix}"
        if src_key_direct in src and tuple(src[src_key_direct].shape) == tuple(dst_template[dst_key].shape):
            out[dst_key] = src[src_key_direct]
        elif src_key_fallback in src and tuple(src[src_key_fallback].shape) == tuple(dst_template[dst_key].shape):
            out[dst_key] = src[src_key_fallback]

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    for k in sorted(set(src.keys()) - set(out.keys())):
        warn.append(f"Ignoring extra source key: {k}")
    return out


def _map_lstm_torch_to_mlx(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    # Copy non-recurrent keys that already match by name/shape.
    for k, dv in dst_template.items():
        if k.startswith("lstm_layers."):
            continue
        sv = src.get(k)
        if sv is not None and tuple(sv.shape) == tuple(dv.shape):
            out[k] = sv

    # Torch -> MLX norm key translation and backward-compat.
    # New torch: lstm.pre_norms.{i}.{weight,bias}, lstm.stack_norm.{weight,bias}
    # Old torch: out_norm.{weight,bias}
    for i in range(n_layers):
        for suffix in ("weight", "bias"):
            dst_pre = f"pre_norms.{i}.{suffix}"
            if dst_pre in dst_template and dst_pre not in out:
                src_new = f"lstm.pre_norms.{i}.{suffix}"
                src_old_pre = f"pre_norm.{suffix}"
                src_old = f"out_norm.{suffix}"
                if src_new in src and tuple(src[src_new].shape) == tuple(dst_template[dst_pre].shape):
                    out[dst_pre] = src[src_new]
                elif src_old_pre in src and tuple(src[src_old_pre].shape) == tuple(dst_template[dst_pre].shape):
                    out[dst_pre] = src[src_old_pre]
                elif src_old in src and tuple(src[src_old].shape) == tuple(dst_template[dst_pre].shape):
                    out[dst_pre] = src[src_old]

    for suffix in ("weight", "bias"):
        src_old = f"out_norm.{suffix}"
        dst_pre = f"pre_norm.{suffix}"
        dst_stack = f"stack_norm.{suffix}"
        src_new_stack = f"lstm.stack_norm.{suffix}"
        if dst_stack in dst_template and dst_stack not in out and src_new_stack in src:
            if tuple(src[src_new_stack].shape) == tuple(dst_template[dst_stack].shape):
                out[dst_stack] = src[src_new_stack]
        if dst_pre in dst_template and dst_pre not in out and src_old in src:
            if tuple(src[src_old].shape) == tuple(dst_template[dst_pre].shape):
                out[dst_pre] = src[src_old]
        if dst_stack in dst_template and dst_stack not in out and src_old in src:
            if tuple(src[src_old].shape) == tuple(dst_template[dst_stack].shape):
                out[dst_stack] = src[src_old]

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
    # Copy non-recurrent keys that already match by name/shape.
    for k, dv in dst_template.items():
        if k.startswith("lstm."):
            continue
        sv = src.get(k)
        if sv is not None and tuple(sv.shape) == tuple(dv.shape):
            out[k] = sv

    # MLX -> Torch norm key translation and backward-compat.
    for i in range(n_layers):
        for suffix in ("weight", "bias"):
            src_pre = f"pre_norms.{i}.{suffix}"
            dst_pre = f"lstm.pre_norms.{i}.{suffix}"
            if dst_pre in dst_template and dst_pre not in out and src_pre in src:
                if tuple(src[src_pre].shape) == tuple(dst_template[dst_pre].shape):
                    out[dst_pre] = src[src_pre]

    for suffix in ("weight", "bias"):
        src_old = f"out_norm.{suffix}"
        src_old_pre = f"pre_norm.{suffix}"
        src_stack = f"stack_norm.{suffix}"
        dst_pre = f"pre_norm.{suffix}"
        dst_stack = f"lstm.stack_norm.{suffix}"
        dst_old = f"out_norm.{suffix}"

        if dst_pre in dst_template and dst_pre not in out and src_old in src:
            if tuple(src[src_old].shape) == tuple(dst_template[dst_pre].shape):
                out[dst_pre] = src[src_old]
        if dst_pre in dst_template and dst_pre not in out and src_old_pre in src:
            if tuple(src[src_old_pre].shape) == tuple(dst_template[dst_pre].shape):
                out[dst_pre] = src[src_old_pre]
        if dst_stack in dst_template and dst_stack not in out:
            src_key = src_stack if src_stack in src else src_old
            if src_key in src and tuple(src[src_key].shape) == tuple(dst_template[dst_stack].shape):
                out[dst_stack] = src[src_key]
        # Support older torch template naming.
        dst_stack_old = "stack_norm." + suffix
        if dst_stack_old in dst_template and dst_stack_old not in out:
            src_key = src_stack if src_stack in src else src_old
            if src_key in src and tuple(src[src_key].shape) == tuple(dst_template[dst_stack_old].shape):
                out[dst_stack_old] = src[src_key]
        if dst_old in dst_template and dst_old not in out:
            src_key = src_stack if src_stack in src else src_old
            if src_key in src and tuple(src[src_key].shape) == tuple(dst_template[dst_old].shape):
                out[dst_old] = src[src_key]

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


def _map_lstm_same_backend(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warn: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, dv in dst_template.items():
        sv = src.get(k)
        if sv is not None and tuple(sv.shape) == tuple(dv.shape):
            out[k] = sv

    for i in range(n_layers):
        for suffix in ("weight", "bias"):
            dst_keys = [
                f"lstm.pre_norms.{i}.{suffix}",
                f"pre_norms.{i}.{suffix}",
            ]
            src_keys = [
                f"lstm.pre_norms.{i}.{suffix}",
                f"pre_norms.{i}.{suffix}",
                f"pre_norm.{suffix}",
                f"out_norm.{suffix}",
            ]
            for dk in dst_keys:
                if dk not in dst_template or dk in out:
                    continue
                for sk in src_keys:
                    if sk in src and tuple(src[sk].shape) == tuple(dst_template[dk].shape):
                        out[dk] = src[sk]
                        break

    for suffix in ("weight", "bias"):
        dst_keys = [f"lstm.stack_norm.{suffix}", f"stack_norm.{suffix}"]
        src_keys = [f"lstm.stack_norm.{suffix}", f"stack_norm.{suffix}", f"out_norm.{suffix}"]
        for dk in dst_keys:
            if dk not in dst_template or dk in out:
                continue
            for sk in src_keys:
                if sk in src and tuple(src[sk].shape) == tuple(dst_template[dk].shape):
                    out[dk] = src[sk]
                    break

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

    if model_name == "neo":
        mapped = _map_neo_state(src_state_np, dst_template, cfg, warn)
    elif model_name == "lstm":
        n_layers = int(cfg.get("n_layers", 1))
        if src_backend == dst_backend:
            mapped = _map_lstm_same_backend(src_state_np, dst_template, n_layers, warn)
        elif src_backend == "torch" and dst_backend == "mlx":
            mapped = _map_lstm_torch_to_mlx(src_state_np, dst_template, n_layers, warn)
        elif src_backend == "mlx" and dst_backend == "torch":
            mapped = _map_lstm_mlx_to_torch(src_state_np, dst_template, n_layers, warn)
        else:
            raise ValueError(f"Unsupported LSTM mapping {src_backend} -> {dst_backend}")
    elif src_backend == dst_backend:
        mapped = _identity_map(src_state_np, dst_template, warn)
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
