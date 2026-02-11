#!/usr/bin/env python3
"""Convert model checkpoints between torch and mlx backends.

This tool converts model weights only (plus lightweight metadata). Optimizer
and scheduler states are intentionally not converted across backends.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from _common import (
    ensure_repo_root_on_path,
    get_backend_api,
    infer_model_name,
    load_yaml,
)


def _to_numpy_state_dict(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


def _torch_template_and_dtype(model) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.dtype]]:
    sd = model.state_dict()
    dtypes = {k: v.dtype for k, v in sd.items()}
    return sd, dtypes


def _mlx_template(model) -> Dict[str, Any]:
    from mlx.utils import tree_flatten

    return {k: v for k, v in tree_flatten(model.parameters())}


def _identity_map(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, dv in dst_template.items():
        if k not in src:
            raise KeyError(f"Missing key '{k}' in source checkpoint.")
        sv = src[k]
        if tuple(sv.shape) != tuple(dv.shape):
            raise ValueError(f"Shape mismatch for key '{k}': src {tuple(sv.shape)} vs dst {tuple(dv.shape)}")
        out[k] = sv
    # Warn on extras
    extras = sorted(set(src.keys()) - set(dst_template.keys()))
    for key in extras:
        warnings.append(f"Ignoring extra source key: {key}")
    return out


def _map_lstm_torch_to_mlx(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    common_keys = [
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
    for k in common_keys:
        if k in dst_template and k in src:
            if tuple(src[k].shape) != tuple(dst_template[k].shape):
                raise ValueError(f"Shape mismatch for key '{k}': src {src[k].shape} vs dst {dst_template[k].shape}")
            out[k] = src[k]

    for i in range(n_layers):
        wx_k = f"lstm_layers.{i}.Wx"
        wh_k = f"lstm_layers.{i}.Wh"
        b_k = f"lstm_layers.{i}.bias"
        w_ih = src[f"lstm.weight_ih_l{i}"]
        w_hh = src[f"lstm.weight_hh_l{i}"]
        b_ih = src[f"lstm.bias_ih_l{i}"]
        b_hh = src[f"lstm.bias_hh_l{i}"]

        if tuple(w_ih.shape) != tuple(dst_template[wx_k].shape):
            raise ValueError(f"Shape mismatch for {wx_k}")
        if tuple(w_hh.shape) != tuple(dst_template[wh_k].shape):
            raise ValueError(f"Shape mismatch for {wh_k}")
        if tuple(b_ih.shape) != tuple(dst_template[b_k].shape):
            raise ValueError(f"Shape mismatch for {b_k}")
        out[wx_k] = w_ih
        out[wh_k] = w_hh
        out[b_k] = b_ih + b_hh

    extras = sorted(set(src.keys()) - set(out.keys()))
    for key in extras:
        warnings.append(f"Ignoring extra source key: {key}")

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def _map_lstm_mlx_to_torch(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    common_keys = [
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
    for k in common_keys:
        if k in dst_template and k in src:
            if tuple(src[k].shape) != tuple(dst_template[k].shape):
                raise ValueError(f"Shape mismatch for key '{k}': src {src[k].shape} vs dst {dst_template[k].shape}")
            out[k] = src[k]

    for i in range(n_layers):
        wx = src[f"lstm_layers.{i}.Wx"]
        wh = src[f"lstm_layers.{i}.Wh"]
        b = src[f"lstm_layers.{i}.bias"]
        out[f"lstm.weight_ih_l{i}"] = wx
        out[f"lstm.weight_hh_l{i}"] = wh
        out[f"lstm.bias_ih_l{i}"] = b
        # Preserve equivalent function by putting recurrent bias into input bias.
        out[f"lstm.bias_hh_l{i}"] = np.zeros_like(b)

    extras = sorted(set(src.keys()) - set(out.keys()))
    for key in extras:
        warnings.append(f"Ignoring extra source key: {key}")
    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def _map_transformer_torch_to_mlx(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    d_model: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    direct_pairs = [
        ("emb.weight", "emb.weight"),
        ("pos_emb.weight", "pos_emb.weight"),
        ("ln_f.weight", "encoder.ln.weight"),
        ("ln_f.bias", "encoder.ln.bias"),
        ("output_bias", "output_bias"),
    ]
    for s, d in direct_pairs:
        if d in dst_template and s in src:
            if tuple(src[s].shape) != tuple(dst_template[d].shape):
                raise ValueError(f"Shape mismatch for '{s}' -> '{d}'")
            out[d] = src[s]

    for i in range(n_layers):
        # qkv split
        qkv_w = src[f"blocks.{i}.attn.qkv.weight"]
        qkv_b = src[f"blocks.{i}.attn.qkv.bias"]
        out[f"encoder.layers.{i}.attention.query_proj.weight"] = qkv_w[0:d_model, :]
        out[f"encoder.layers.{i}.attention.key_proj.weight"] = qkv_w[d_model : 2 * d_model, :]
        out[f"encoder.layers.{i}.attention.value_proj.weight"] = qkv_w[2 * d_model : 3 * d_model, :]
        # MLX attention path has no qkv/out biases in current model.
        warnings.append(f"Dropping torch attention biases for layer {i} (MLX transformer has no attention bias params).")

        out[f"encoder.layers.{i}.attention.out_proj.weight"] = src[f"blocks.{i}.attn.out_proj.weight"]
        out[f"encoder.layers.{i}.ln1.weight"] = src[f"blocks.{i}.ln1.weight"]
        out[f"encoder.layers.{i}.ln1.bias"] = src[f"blocks.{i}.ln1.bias"]
        out[f"encoder.layers.{i}.ln2.weight"] = src[f"blocks.{i}.ln2.weight"]
        out[f"encoder.layers.{i}.ln2.bias"] = src[f"blocks.{i}.ln2.bias"]
        out[f"encoder.layers.{i}.linear1.weight"] = src[f"blocks.{i}.mlp.0.weight"]
        out[f"encoder.layers.{i}.linear1.bias"] = src[f"blocks.{i}.mlp.0.bias"]
        out[f"encoder.layers.{i}.linear2.weight"] = src[f"blocks.{i}.mlp.3.weight"]
        out[f"encoder.layers.{i}.linear2.bias"] = src[f"blocks.{i}.mlp.3.bias"]

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def _map_transformer_mlx_to_torch(
    src: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    n_layers: int,
    d_model: int,
    warnings: List[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    direct_pairs = [
        ("emb.weight", "emb.weight"),
        ("pos_emb.weight", "pos_emb.weight"),
        ("encoder.ln.weight", "ln_f.weight"),
        ("encoder.ln.bias", "ln_f.bias"),
        ("output_bias", "output_bias"),
    ]
    for s, d in direct_pairs:
        if d in dst_template and s in src:
            if tuple(src[s].shape) != tuple(dst_template[d].shape):
                raise ValueError(f"Shape mismatch for '{s}' -> '{d}'")
            out[d] = src[s]

    for i in range(n_layers):
        qw = src[f"encoder.layers.{i}.attention.query_proj.weight"]
        kw = src[f"encoder.layers.{i}.attention.key_proj.weight"]
        vw = src[f"encoder.layers.{i}.attention.value_proj.weight"]
        qkv_w = np.concatenate([qw, kw, vw], axis=0)
        out[f"blocks.{i}.attn.qkv.weight"] = qkv_w
        out[f"blocks.{i}.attn.qkv.bias"] = np.zeros((3 * d_model,), dtype=qw.dtype)
        out[f"blocks.{i}.attn.out_proj.weight"] = src[f"encoder.layers.{i}.attention.out_proj.weight"]
        out[f"blocks.{i}.attn.out_proj.bias"] = np.zeros((d_model,), dtype=qw.dtype)
        warnings.append(f"Filling torch attention biases with zeros for layer {i} (MLX source has no attention bias params).")

        out[f"blocks.{i}.ln1.weight"] = src[f"encoder.layers.{i}.ln1.weight"]
        out[f"blocks.{i}.ln1.bias"] = src[f"encoder.layers.{i}.ln1.bias"]
        out[f"blocks.{i}.ln2.weight"] = src[f"encoder.layers.{i}.ln2.weight"]
        out[f"blocks.{i}.ln2.bias"] = src[f"encoder.layers.{i}.ln2.bias"]
        out[f"blocks.{i}.mlp.0.weight"] = src[f"encoder.layers.{i}.linear1.weight"]
        out[f"blocks.{i}.mlp.0.bias"] = src[f"encoder.layers.{i}.linear1.bias"]
        out[f"blocks.{i}.mlp.3.weight"] = src[f"encoder.layers.{i}.linear2.weight"]
        out[f"blocks.{i}.mlp.3.bias"] = src[f"encoder.layers.{i}.linear2.bias"]

    missing = sorted(set(dst_template.keys()) - set(out.keys()))
    if missing:
        raise KeyError(f"Missing mapped destination keys: {missing}")
    return out


def _map_model_state(
    model_name: str,
    src_backend: str,
    dst_backend: str,
    src_state: Dict[str, np.ndarray],
    dst_template: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    warnings: List[str] = []
    if src_backend == dst_backend:
        return _identity_map(src_state, dst_template, warnings), warnings

    if model_name == "neo":
        # Neo key layout is backend-aligned in this repo.
        return _identity_map(src_state, dst_template, warnings), warnings

    if model_name == "lstm":
        n_layers = int(cfg.get("n_layers", 1))
        if src_backend == "torch" and dst_backend == "mlx":
            return _map_lstm_torch_to_mlx(src_state, dst_template, n_layers, warnings), warnings
        if src_backend == "mlx" and dst_backend == "torch":
            return _map_lstm_mlx_to_torch(src_state, dst_template, n_layers, warnings), warnings

    if model_name == "transformer":
        n_layers = int(cfg.get("n_layers", 1))
        d_model = int(cfg["d_model"])
        if src_backend == "torch" and dst_backend == "mlx":
            return _map_transformer_torch_to_mlx(src_state, dst_template, n_layers, d_model, warnings), warnings
        if src_backend == "mlx" and dst_backend == "torch":
            return _map_transformer_mlx_to_torch(src_state, dst_template, n_layers, d_model, warnings), warnings

    raise ValueError(f"Unsupported conversion path: {src_backend} -> {dst_backend} for model {model_name}")


def _infer_source_backend(payload: Dict[str, Any]) -> str:
    if payload.get("backend") == "mlx":
        return "mlx"
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict):
        for v in state.values():
            if isinstance(v, torch.Tensor):
                return "torch"
    # Fallback to torch-style unless explicitly marked.
    return "torch"


def _load_source_checkpoint(path: Path, src_backend: str | None) -> Tuple[str, Dict[str, Any], Dict[str, np.ndarray]]:
    # Try torch load first for torch payloads.
    payload: Dict[str, Any]
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            payload = obj
        else:
            payload = {"model_state_dict": obj}
    except Exception:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, dict):
            raise ValueError("Checkpoint payload must be a dict-like object.")
        payload = obj

    detected = _infer_source_backend(payload)
    if src_backend is None:
        src_backend = detected
    if src_backend not in ("torch", "mlx"):
        raise ValueError(f"Unsupported source backend '{src_backend}'.")
    if detected != src_backend:
        raise ValueError(f"Source backend mismatch: requested '{src_backend}' but checkpoint looks like '{detected}'.")

    raw_state = payload.get("model_state_dict")
    if raw_state is None:
        if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in payload.values()):
            raw_state = payload
            payload = {"model_state_dict": raw_state}
        else:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    if not isinstance(raw_state, dict):
        raise ValueError("'model_state_dict' must be a dict.")

    return src_backend, payload, _to_numpy_state_dict(raw_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoints between torch and mlx backends.")
    parser.add_argument("--config", required=True, help="Path to config YAML (target model shape source of truth).")
    parser.add_argument("--input", required=True, help="Input checkpoint path.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--dst-backend", required=True, choices=["torch", "mlx"], help="Destination backend.")
    parser.add_argument("--src-backend", default="auto", choices=["auto", "torch", "mlx"], help="Source backend.")
    parser.add_argument("--model-name", default=None, help="Override model name (neo|lstm|transformer).")
    args = parser.parse_args()

    ensure_repo_root_on_path()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    model_name = args.model_name.strip().lower() if args.model_name else infer_model_name(cfg_path, cfg)
    if model_name not in ("neo", "lstm", "transformer"):
        raise ValueError(f"Unsupported model '{model_name}'.")

    src_backend_req = None if args.src_backend == "auto" else args.src_backend
    src_backend, src_payload, src_state = _load_source_checkpoint(Path(args.input).expanduser().resolve(), src_backend_req)
    dst_backend = args.dst_backend

    dst_api = get_backend_api(dst_backend)
    dst_model = dst_api.build_model(cfg, model_name)
    if dst_backend == "torch":
        dst_template, dst_dtypes = _torch_template_and_dtype(dst_model)
    else:
        dst_template = _mlx_template(dst_model)
        dst_dtypes = {}

    converted_state, warnings = _map_model_state(
        model_name=model_name,
        src_backend=src_backend,
        dst_backend=dst_backend,
        src_state=src_state,
        dst_template=dst_template,
        cfg=cfg,
    )

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_backend == "torch":
        torch_state = {}
        for k, dv in dst_template.items():
            arr = converted_state[k]
            t = torch.from_numpy(np.asarray(arr))
            t = t.to(dtype=dst_dtypes[k])
            torch_state[k] = t
            if tuple(t.shape) != tuple(dv.shape):
                raise ValueError(f"Converted shape mismatch for key '{k}': {tuple(t.shape)} vs {tuple(dv.shape)}")
        out_payload = {
            "model_state_dict": torch_state,
            "epoch": int(src_payload.get("epoch", 0)),
            "global_step": int(src_payload.get("global_step", 0)),
            "cfg": dict(src_payload.get("cfg", cfg)),
        }
        if "best_val" in src_payload:
            out_payload["best_val"] = float(src_payload["best_val"])
        torch.save(out_payload, str(out_path))
    else:
        out_payload = {
            "backend": "mlx",
            "model_state_dict": {k: np.asarray(v) for k, v in converted_state.items()},
            "epoch": int(src_payload.get("epoch", 0)),
            "global_step": int(src_payload.get("global_step", 0)),
            "cfg": dict(src_payload.get("cfg", cfg)),
        }
        if "best_val" in src_payload:
            out_payload["best_val"] = float(src_payload["best_val"])
        with out_path.open("wb") as handle:
            pickle.dump(out_payload, handle)

    print(f"Converted checkpoint: {src_backend} -> {dst_backend}")
    print(f"Model: {model_name}")
    print(f"Input:  {args.input}")
    print(f"Output: {out_path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
