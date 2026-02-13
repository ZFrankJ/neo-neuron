"""MLX backend adapter."""

from __future__ import annotations

import gc
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as mxnn
    import mlx.optimizers as mxoptim
    from mlx.utils import tree_flatten, tree_unflatten
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("MLX backend requested but 'mlx' is not installed.") from exc

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from ..checkpoint_compat import (
    infer_checkpoint_backend,
    infer_model_name_from_model,
    load_checkpoint_payload,
    map_model_state,
    to_numpy_state_dict,
)

NAME = "mlx"


def _cfg_get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_betas(value: Any, default=(0.9, 0.95)) -> Tuple[float, float]:
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        if isinstance(value, str):
            a, b = [p.strip() for p in value.split(",", 1)]
            return (float(a), float(b))
    except (TypeError, ValueError):
        return default
    return default


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Missing required config key: '{key}'")
    return cfg[key]


def _resolve_recurrent_norm(cfg: Dict[str, Any]) -> str:
    if cfg.get("recurrent_norm") not in (None, ""):
        return str(cfg["recurrent_norm"])
    if cfg.get("output_norm") not in (None, ""):
        print(
            "Warning: 'output_norm' is deprecated; use 'recurrent_norm'. "
            "Falling back to 'output_norm' for compatibility.",
            file=sys.stderr,
            flush=True,
        )
        return str(cfg["output_norm"])
    print(
        "Warning: neither 'recurrent_norm' nor deprecated 'output_norm' is set; "
        "defaulting to 'layernorm'.",
        file=sys.stderr,
        flush=True,
    )
    return "layernorm"


def _flatten_tree(tree: Any) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in tree_flatten(tree)}


def _unflatten_tree(flat: Dict[str, np.ndarray]) -> Any:
    return tree_unflatten([(k, mx.array(v)) for k, v in flat.items()])


def _log_line(message: str) -> None:
    print(message, flush=True)


def _mem_report(tag: str = "") -> None:
    if psutil is None:
        return
    rss = psutil.Process().memory_info().rss / (1024 ** 3)
    try:
        if not hasattr(mx, "get_active_memory"):
            _log_line(f"{tag} | RSS={rss:.2f} GB")
            return
        active = mx.get_active_memory() / (1024 ** 3)
        _log_line(f"{tag} | RSS={rss:.2f} GB | MLX={active:.2f} GB")
    except Exception:
        _log_line(f"{tag} | RSS={rss:.2f} GB")


def _maybe_report_memory(step: int, interval: Optional[int]) -> None:
    if interval is None or interval <= 0:
        return
    if step % interval == 0:
        _mem_report(tag=f"step {step}")


def _clear_memory() -> None:
    gc.collect()
    try:
        # New MLX API; keep compatibility with older releases.
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def _maybe_clear_memory(step: int, interval: Optional[int]) -> None:
    if interval is None or interval <= 0:
        return
    if step % interval == 0:
        _clear_memory()


def _count_params_tree(tree: Dict[str, Any]) -> int:
    total = 0
    for _, arr in tree_flatten(tree):
        total += int(np.prod(arr.shape))
    return total


def count_params(model) -> int:
    return _count_params_tree(model.parameters())


def _param_breakdown(model_name: str, model) -> Dict[str, int]:
    flat = dict(tree_flatten(model.parameters()))
    embeddings = 0
    recurrent = 0
    transformer = 0
    proj_head = 0
    for name, arr in flat.items():
        n = int(np.prod(arr.shape))
        if ".emb." in name or name.startswith("emb."):
            embeddings += n
            continue
        if "pos_emb" in name:
            embeddings += n
            continue
        if model_name == "neo" and "recurrent." in name:
            recurrent += n
            continue
        if model_name == "lstm" and (
            "lstm_layers." in name or name.startswith("pre_norms.") or name.startswith("stack_norm.")
        ):
            recurrent += n
            continue
        if model_name == "transformer" and "encoder." in name:
            transformer += n
            continue
        proj_head += n
    out = {"embeddings": embeddings}
    if model_name == "transformer":
        out["transformer"] = transformer
        out["head"] = proj_head
    else:
        out["recurrent"] = recurrent
        out["proj_head"] = proj_head
    return out


def _parse_activation_id(activation_id) -> int:
    if isinstance(activation_id, str):
        text = activation_id.strip().lower()
        if text.startswith("id"):
            text = text[2:]
        value = int(text)
    else:
        value = int(activation_id)
    if value not in (3, 4, 5):
        raise ValueError(f"Unsupported activation_id '{activation_id}'. Expected id3/id4/id5.")
    return value


def _negative_branch(x: mx.array, activation_id: int) -> mx.array:
    x2 = x * x
    x3 = x * x2
    if activation_id == 4:
        x4 = x2 * x2
        return (x4 / 6.0) + (x3 / 6.0) - x2 + x
    if activation_id == 5:
        x4 = x2 * x2
        x5 = x * x4
        return (x5 / 120.0) + (x4 / 6.0) + (x3 / 6.0) - x2 + x
    return (x3 / 6.0) - x2 + x


def _cortical_activation(x: mx.array, activation_id: int) -> mx.array:
    pos = mx.tanh(x)
    neg = _negative_branch(x, activation_id) * mx.exp(mx.minimum(x, 0.0))
    return mx.where(x >= 0, pos, neg)


def _build_output_norm(norm_type: str, dims: int) -> mxnn.Module:
    norm = str(norm_type).strip().lower()
    if norm in ("none", "off", "identity"):
        return mxnn.Identity()
    if norm in ("layernorm", "layer_norm", "ln"):
        return mxnn.LayerNorm(dims)
    if norm in ("rmsnorm", "rms_norm", "rms"):
        return mxnn.RMSNorm(dims)
    raise ValueError(f"Unsupported output_norm '{norm_type}'.")


class MlxCorticalNeuron(mxnn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation_id="id3"):
        super().__init__()
        self.fg_linear = mxnn.Linear(input_dim, 2 * output_dim)
        self.activation_id = _parse_activation_id(activation_id)

    def __call__(self, x: mx.array, prev_state: mx.array):
        fg = self.fg_linear(x)
        f_x_raw, g_x_raw = mx.split(fg, 2, axis=-1)
        hidden = prev_state + f_x_raw
        state = _cortical_activation(hidden, self.activation_id)
        output = state * g_x_raw
        aux = None if self.training else {"f_x_raw": f_x_raw, "g_x_raw": g_x_raw}
        return output, state, aux


class MlxCorticalRecurrentStack(mxnn.Module):
    def __init__(self, d_model: int, n_layers: int, cell_kwargs: Dict[str, Any], output_norm: str = "layernorm"):
        super().__init__()
        cell_kwargs = dict(cell_kwargs or {})
        # Norm is now handled at stack level (pre-layer + final norm), not in the cell.
        cell_kwargs.pop("output_norm", None)
        self.layers = [MlxCorticalNeuron(d_model, d_model, **cell_kwargs) for _ in range(n_layers)]
        self.pre_norms = [_build_output_norm(output_norm, d_model) for _ in range(n_layers)]
        self.stack_norm = _build_output_norm(output_norm, d_model)
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)

    def init_state(self, batch_size: int) -> mx.array:
        return mx.zeros((self.n_layers, batch_size, self.d_model), dtype=mx.float32)

    def __call__(self, x: mx.array, state: Optional[mx.array] = None):
        t, b, d = x.shape
        if state is None:
            state = self.init_state(b)
        states = [state[i] for i in range(self.n_layers)]
        ys = []
        for ti in range(t):
            h = x[ti]
            for li, layer in enumerate(self.layers):
                h_norm = self.pre_norms[li](h)
                h, ns, _ = layer(h_norm, states[li])
                states[li] = ns
            h = self.stack_norm(h)
            ys.append(h)
        y = mx.stack(ys, axis=0)
        new_state = mx.stack(states, axis=0)
        return y, new_state


class MlxNeoLM(mxnn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_embed: int,
        n_layers: int,
        dropout: float,
        tie_embeddings: bool,
        cell_kwargs: Dict[str, Any],
        output_norm: str = "layernorm",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_embed = int(d_embed)
        self.n_layers = int(n_layers)
        self.tie_embeddings = bool(tie_embeddings)

        self.emb = mxnn.Embedding(vocab_size, d_embed)
        self.in_proj = mxnn.Linear(d_embed, d_model) if d_embed != d_model else mxnn.Identity()
        self.recurrent = MlxCorticalRecurrentStack(d_model, n_layers, cell_kwargs, output_norm=output_norm)
        self.drop = mxnn.Dropout(dropout)
        self.out_proj = mxnn.Linear(d_model, d_embed) if d_embed != d_model else mxnn.Identity()
        if self.tie_embeddings:
            self.output_bias = mx.zeros((vocab_size,), dtype=mx.float32)
            self.head = None
        else:
            self.head = mxnn.Linear(d_model, vocab_size)
            self.output_bias = None

    def init_state(self, batch_size: int):
        return self.recurrent.init_state(batch_size)

    def __call__(self, idx: mx.array, state: Optional[mx.array]):
        x = self.emb(idx)  # [T, B, E]
        x = self.in_proj(x)
        y, new_state = self.recurrent(x, state)
        y = self.drop(y)
        if self.tie_embeddings:
            y = self.out_proj(y)
            logits = mx.matmul(y, self.emb.weight.T)
            logits = logits + self.output_bias
        else:
            logits = self.head(y)
        return logits, new_state


class MlxLSTMLM(mxnn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_embed: int,
        n_layers: int,
        dropout: float,
        tie_embeddings: bool,
        output_norm: str = "layernorm",
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_embed = int(d_embed)
        self.n_layers = int(n_layers)
        self.tie_embeddings = bool(tie_embeddings)

        self.emb = mxnn.Embedding(vocab_size, d_embed)
        self.in_proj = mxnn.Linear(d_embed, d_model) if d_embed != d_model else mxnn.Identity()
        self.lstm_layers = [mxnn.LSTM(d_model, d_model) for _ in range(n_layers)]
        self.pre_norms = [_build_output_norm(output_norm, d_model) for _ in range(n_layers)]
        self.stack_norm = _build_output_norm(output_norm, d_model)
        self.drop = mxnn.Dropout(dropout)
        self.layer_drop = float(dropout)
        self.out_proj = mxnn.Linear(d_model, d_embed) if d_embed != d_model else mxnn.Identity()
        if self.tie_embeddings:
            self.output_bias = mx.zeros((vocab_size,), dtype=mx.float32)
            self.head = None
        else:
            self.head = mxnn.Linear(d_model, vocab_size)
            self.output_bias = None

    def init_state(self, batch_size: int):
        h = mx.zeros((self.n_layers, batch_size, self.d_model), dtype=mx.float32)
        c = mx.zeros((self.n_layers, batch_size, self.d_model), dtype=mx.float32)
        return (h, c)

    def __call__(self, idx: mx.array, state: Optional[Tuple[mx.array, mx.array]]):
        x = self.emb(idx)  # [T, B, E]
        x = self.in_proj(x)
        h = mx.swapaxes(x, 0, 1)  # [B, T, D]

        if state is None:
            hs, cs = None, None
        else:
            hs, cs = state

        new_hs: List[mx.array] = []
        new_cs: List[mx.array] = []
        for li, layer in enumerate(self.lstm_layers):
            h = self.pre_norms[li](h)
            h0 = None if hs is None else hs[li]
            c0 = None if cs is None else cs[li]
            if h0 is None:
                h_seq, c_seq = layer(h)
            else:
                h_seq, c_seq = layer(h, h0, c0)
            h = h_seq
            if self.layer_drop > 0.0 and self.training and li < (self.n_layers - 1):
                h = self.drop(h)
            new_hs.append(h_seq[:, -1, :])
            new_cs.append(c_seq[:, -1, :])

        y = mx.swapaxes(h, 0, 1)  # [T, B, D]
        y = self.stack_norm(y)
        y = self.drop(y)
        if self.tie_embeddings:
            y = self.out_proj(y)
            logits = mx.matmul(y, self.emb.weight.T)
            logits = logits + self.output_bias
        else:
            logits = self.head(y)
        new_state = (mx.stack(new_hs, axis=0), mx.stack(new_cs, axis=0))
        return logits, new_state


class MlxTransformerLM(mxnn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        tie_embeddings: bool,
        max_seq_len: int,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ff_mult = int(ff_mult)
        self.tie_embeddings = bool(tie_embeddings)

        self.emb = mxnn.Embedding(vocab_size, d_model)
        self.pos_emb = mxnn.Embedding(max_seq_len, d_model)
        self.drop = mxnn.Dropout(dropout)
        self.encoder = mxnn.TransformerEncoder(
            num_layers=n_layers,
            dims=d_model,
            num_heads=n_heads,
            mlp_dims=ff_mult * d_model,
            dropout=dropout,
            norm_first=True,
        )
        if self.tie_embeddings:
            self.output_bias = mx.zeros((vocab_size,), dtype=mx.float32)
            self.head = None
        else:
            self.head = mxnn.Linear(d_model, vocab_size)
            self.output_bias = None

    def __call__(self, idx: mx.array, state=None):
        t, _ = idx.shape
        idx_bt = mx.swapaxes(idx, 0, 1)  # [B, T]
        pos = mx.arange(t, dtype=mx.int32)
        pos_emb = self.pos_emb(pos)[None, :, :]
        x = self.emb(idx_bt) + pos_emb
        x = self.drop(x)
        mask = mx.triu(mx.ones((t, t), dtype=mx.bool_), k=1)
        x = self.encoder(x, mask)

        if self.tie_embeddings:
            logits_bt = mx.matmul(x, self.emb.weight.T) + self.output_bias
        else:
            logits_bt = self.head(x)
        logits = mx.swapaxes(logits_bt, 0, 1)
        return logits, state


def build_model(cfg: Dict[str, Any], model_name: str):
    vocab_size = int(_require(cfg, "vocab_size"))
    d_model = int(_require(cfg, "d_model"))
    d_embed = int(cfg.get("d_embed", d_model))
    n_layers = int(cfg.get("n_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    tie_embeddings = bool(cfg.get("tie_embeddings", True))
    recurrent_norm = _resolve_recurrent_norm(cfg)

    if model_name == "neo":
        cell_kwargs = {
            "activation_id": cfg.get("activation_id", "id3"),
        }
        return MlxNeoLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            cell_kwargs=cell_kwargs,
            output_norm=recurrent_norm,
        )
    if model_name == "lstm":
        return MlxLSTMLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            output_norm=recurrent_norm,
        )
    if model_name == "transformer":
        n_heads = int(_require(cfg, "n_heads"))
        return MlxTransformerLM(
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


def supports_probe() -> bool:
    return False


def get_runtime_device(requested: Optional[str] = None):
    choice = "gpu"
    if requested is not None and requested != "auto":
        req = str(requested).strip().lower()
        if req in ("cpu",):
            mx.set_default_device(mx.Device(mx.cpu, 0))
            choice = "cpu"
        elif req in ("gpu", "mps", "cuda"):
            mx.set_default_device(mx.Device(mx.gpu, 0))
            choice = "gpu"
        else:
            raise ValueError(f"Unsupported MLX device '{requested}'. Use cpu|mps|gpu|auto.")
    else:
        mx.set_default_device(mx.Device(mx.gpu, 0))
    return choice


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _to_numpy_ids(ids) -> np.ndarray:
    try:
        import torch

        if isinstance(ids, torch.Tensor):
            return ids.detach().cpu().numpy().astype(np.int32, copy=False)
    except Exception:
        pass
    arr = np.asarray(ids)
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)
    return arr


def _validate_token_id_range(cfg: Any, *arrays: np.ndarray, context: str = "mlx") -> None:
    vocab_size = int(_cfg_get(cfg, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise ValueError(f"[{context}] Invalid vocab_size={vocab_size}. Set a positive vocab_size in config.")
    max_id = -1
    for arr in arrays:
        if arr is None or arr.size == 0:
            continue
        cur = int(arr.max())
        if cur > max_id:
            max_id = cur
    if max_id >= vocab_size:
        raise ValueError(
            f"[{context}] token id range exceeds config vocab_size: max_token_id={max_id}, "
            f"requires vocab_size >= {max_id + 1}, but config has vocab_size={vocab_size}. "
            "This can cause embedding OOB and Metal/MLX crashes."
        )


def _random_batch(ids: np.ndarray, batch_size: int, block_size: int) -> Tuple[mx.array, mx.array]:
    n_tokens = int(ids.shape[0])
    max_start = max(1, n_tokens - block_size - 1)
    starts = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(block_size + 1, dtype=np.int64)
    seq = ids[starts[:, None] + offsets[None, :]]  # [B, T+1]
    x_np = np.ascontiguousarray(seq[:, :-1].T)
    y_np = np.ascontiguousarray(seq[:, 1:].T)
    return mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32)


def _streaming_batches(ids: np.ndarray, block_size: int, batch_size: int):
    n_tokens = int(ids.shape[0])
    step = block_size * batch_size
    for start in range(0, n_tokens - (block_size + 1), step):
        cur_b = min(batch_size, (n_tokens - (start + block_size + 1)) // block_size)
        if cur_b <= 0:
            break
        x = np.stack([ids[start + i * block_size: start + i * block_size + block_size] for i in range(cur_b)])
        y = np.stack([ids[start + i * block_size + 1: start + i * block_size + block_size + 1] for i in range(cur_b)])
        yield mx.array(np.ascontiguousarray(x.T), dtype=mx.int32), mx.array(np.ascontiguousarray(y.T), dtype=mx.int32), cur_b


def _init_state_for_model(model, batch_size: int):
    if hasattr(model, "init_state"):
        return model.init_state(batch_size)
    return None


def _stop_grad_tree(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_stop_grad_tree(v) for v in x)
    if isinstance(x, list):
        return [_stop_grad_tree(v) for v in x]
    return mx.stop_gradient(x)


def _state_batch_size(state) -> Optional[int]:
    if state is None:
        return None
    if isinstance(state, tuple) and state:
        first = state[0]
        if hasattr(first, "shape") and len(first.shape) >= 2:
            return int(first.shape[1])
        return None
    if hasattr(state, "shape") and len(state.shape) >= 2:
        return int(state.shape[1])
    return None


@dataclass
class _LRScheduler:
    cosine: bool
    base_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def lr(self, step: int) -> float:
        if not self.cosine:
            return self.base_lr
        if step < self.warmup_steps:
            return self.base_lr * (step / float(max(1, self.warmup_steps)))
        progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


def _build_scheduler(cfg: Any, steps_per_epoch: int) -> _LRScheduler:
    cosine = bool(_cfg_get(cfg, "cosine", True))
    epochs = int(_cfg_get(cfg, "epochs", 1))
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_epochs = float(_cfg_get(cfg, "warmup_epochs", 0))
    warmup_steps = max(1, int(warmup_epochs * max(1, steps_per_epoch)))
    base_lr = float(_cfg_get(cfg, "lr", 3e-4))
    min_lr = float(_cfg_get(cfg, "min_lr", 0.0))
    return _LRScheduler(
        cosine=cosine,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )


def _build_weight_decay_lookup(model, model_name: str, cfg: Any) -> Dict[str, float]:
    policy = str(_cfg_get(cfg, "weight_decay_policy", "table")).lower()
    embed_wd = _to_float(_cfg_get(cfg, "embed_weight_decay", 0.0), 0.0)
    proj_wd = _to_float(_cfg_get(cfg, "proj_weight_decay", 1e-3), 1e-3)
    recurrent_wd = _to_float(_cfg_get(cfg, "recurrent_weight_decay", 0.0), 0.0)
    transformer_wd = _to_float(_cfg_get(cfg, "transformer_weight_decay", 1e-2), 1e-2)
    uniform_wd = _to_float(_cfg_get(cfg, "weight_decay", 0.0), 0.0)
    flat = dict(tree_flatten(model.parameters()))
    out: Dict[str, float] = {}

    for name in flat:
        leaf = name.split(".")[-1]
        if name.endswith("output_bias") or leaf == "bias" or leaf.startswith("bias_"):
            out[name] = 0.0
            continue
        if ".out_norm." in name or ".ln." in name or ".ln1." in name or ".ln2." in name or "norm." in name or name.startswith("pre_norms."):
            out[name] = 0.0
            continue
        if ".emb." in name or name.startswith("emb.") or "pos_emb." in name:
            out[name] = 0.0 if policy not in ("table", "per_param", "per-parameter") else embed_wd
            continue

        if policy in ("table", "per_param", "per-parameter"):
            if model_name == "transformer":
                out[name] = transformer_wd
            elif model_name == "neo":
                out[name] = recurrent_wd if name.startswith("recurrent.") else proj_wd
            elif model_name == "lstm":
                out[name] = recurrent_wd if name.startswith("lstm_layers.") else proj_wd
            else:
                out[name] = proj_wd
        else:
            out[name] = uniform_wd
    return out


def _apply_decoupled_weight_decay(model, wd_lookup: Dict[str, float], lr: float) -> None:
    params = model.parameters()
    flat = dict(tree_flatten(params))
    changed = False
    for name, arr in flat.items():
        wd = float(wd_lookup.get(name, 0.0))
        if wd != 0.0:
            flat[name] = arr * (1.0 - lr * wd)
            changed = True
    if changed:
        model.update(tree_unflatten(list(flat.items())))


def _write_progress(run_dir: Optional[str], payload: Dict[str, float]) -> None:
    if not run_dir:
        return
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "progress.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2, sort_keys=True)
    with (path / "history.jsonl").open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(payload) + "\n")


def save_checkpoint_entry(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    cfg: Any,
    best_val: Optional[float] = None,
) -> None:
    payload = {
        "format": "neo_unified_checkpoint_v1",
        "backend": "mlx",
        "model_state_dict": _flatten_tree(model.parameters()),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "cfg": _cfg_to_dict(cfg),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = _flatten_tree(optimizer.state)
    if scheduler is not None:
        payload["scheduler_state_dict"] = dict(scheduler)
    if best_val is not None:
        payload["best_val"] = float(best_val)
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint_entry(path: str | Path, model, optimizer=None, scheduler=None, device=None) -> Dict[str, Any]:
    del device
    ckpt = load_checkpoint_payload(path, map_location="cpu")
    raw_state = ckpt.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("Checkpoint missing 'model_state_dict' mapping.")

    src_backend = infer_checkpoint_backend(ckpt)
    model_name = infer_model_name_from_model(model)
    dst_template = dict(tree_flatten(model.parameters()))
    mapped_state, _ = map_model_state(
        model_name=model_name,
        src_backend=src_backend,
        dst_backend="mlx",
        src_state_np=to_numpy_state_dict(raw_state),
        dst_template=dst_template,
        cfg=ckpt.get("cfg", {}),
    )
    model.update(_unflatten_tree(mapped_state))
    if src_backend == "mlx":
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.state = _unflatten_tree(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.update(ckpt["scheduler_state_dict"])
    return ckpt


def _eval_perplexity(model, ids: np.ndarray, cfg: Any) -> float:
    model.eval()
    t_len = int(_cfg_get(cfg, "block_size", 128))
    batch = int(_cfg_get(cfg, "batch_size", 16))
    total_loss = 0.0
    total_tokens = 0
    n = int(ids.shape[0])

    for start in range(0, n - (t_len + 1), t_len * batch):
        cur_b = min(batch, (n - (start + t_len + 1)) // t_len)
        if cur_b <= 0:
            break
        x = np.stack([ids[start + i * t_len: start + i * t_len + t_len] for i in range(cur_b)])
        y = np.stack([ids[start + i * t_len + 1: start + i * t_len + t_len + 1] for i in range(cur_b)])
        x_mx = mx.array(np.ascontiguousarray(x.T), dtype=mx.int32)
        y_mx = mx.array(np.ascontiguousarray(y.T), dtype=mx.int32)
        state = _init_state_for_model(model, cur_b)
        logits, _ = model(x_mx, state)
        loss = mxnn.losses.cross_entropy(logits, y_mx, reduction="sum")
        mx.eval(loss)
        total_loss += float(loss.item())
        total_tokens += cur_b * t_len
    return math.exp(total_loss / max(1, total_tokens))


def eval_metrics_entry(model, ids, cfg: Any, device=None):
    del device
    ids_np = _to_numpy_ids(ids)
    _validate_token_id_range(cfg, ids_np, context="mlx_eval")
    ppl = _eval_perplexity(model, ids_np, cfg)
    return {"ppl": ppl, "act_sparsity": None, "gflops_per_token": None}


def train_entry(
    model,
    cfg: Any,
    train_ids,
    val_ids,
    test_ids=None,
    device=None,
):
    del device
    model_name = str(_cfg_get(cfg, "model_name", "unknown")).strip().lower()
    total_params = count_params(model)
    breakdown = _param_breakdown(model_name, model)
    _log_line("== Model Info ==")
    _log_line(f"Model tag: {model_name}")
    _log_line(f"Model: {model.__class__.__name__}")
    _log_line(f"Total params: {total_params / 1e6:.2f}M")
    if breakdown:
        parts = [f"{k}={v / 1e6:.2f}M" for k, v in breakdown.items()]
        _log_line("Breakdown: " + " | ".join(parts))

    train_np = _to_numpy_ids(train_ids)
    val_np = _to_numpy_ids(val_ids)
    test_np = _to_numpy_ids(test_ids) if test_ids is not None else None
    _validate_token_id_range(cfg, train_np, val_np, test_np, context="mlx_train")

    lr = _to_float(_cfg_get(cfg, "lr", 3e-4), 3e-4)
    betas = _to_betas(_cfg_get(cfg, "betas", (0.9, 0.95)), (0.9, 0.95))
    eps = _to_float(_cfg_get(cfg, "adam_eps", 1e-8), 1e-8)
    optimizer = mxoptim.AdamW(learning_rate=lr, betas=list(betas), eps=eps, weight_decay=0.0)

    batch_size = int(_cfg_get(cfg, "batch_size", 1))
    block_size = int(_cfg_get(cfg, "block_size", 1))
    steps_per_epoch = max(1, train_np.shape[0] // (batch_size * block_size))
    scheduler = _build_scheduler(cfg, steps_per_epoch)
    wd_lookup = _build_weight_decay_lookup(model, model_name, cfg)

    save_dir = str(_cfg_get(cfg, "save_dir", "checkpoints"))
    run_tag = str(_cfg_get(cfg, "run_tag", "run"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_path = str(Path(save_dir) / f"best_{run_tag}.pt")
    last_path = str(Path(save_dir) / f"last_{run_tag}.pt")

    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    resume_path = str(_cfg_get(cfg, "resume_path", "") or "")
    if resume_path:
        ckpt = load_checkpoint_entry(resume_path, model, optimizer=optimizer)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        _log_line(f"Resumed from {resume_path} (epoch {start_epoch - 1}, global_step {global_step})")
        # Avoid retaining full checkpoint payload (including optimizer state arrays).
        del ckpt
        _clear_memory()

    epochs = int(_cfg_get(cfg, "epochs", 1))
    grad_clip = _to_float(_cfg_get(cfg, "grad_clip", 1.0), 1.0)
    mem_interval = _cfg_get(cfg, "mem_report_interval", None)
    mem_clear_interval = _cfg_get(cfg, "mem_clear_interval", None)

    train_regime = str(_cfg_get(cfg, "train_regime", "random")).lower()
    stream_state = bool(_cfg_get(cfg, "stream_state", False))
    if train_regime not in ("random", "streaming"):
        raise ValueError(f"Unsupported train_regime '{train_regime}'.")

    run_dir = _cfg_get(cfg, "run_dir", None)
    vocab_size = int(_cfg_get(cfg, "vocab_size", 0))
    tbptt_len = int(_cfg_get(cfg, "tbptt_len", 0))

    def _loss_fn(x_batch: mx.array, y_batch: mx.array, state):
        logits, new_state = model(x_batch, state)
        loss = mxnn.losses.cross_entropy(logits, y_batch, reduction="mean")
        return loss, new_state

    loss_and_grad = mxnn.value_and_grad(model, _loss_fn)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        if train_regime == "random":
            step_iter: Iterable[Any] = range(steps_per_epoch)
            state = None
        else:
            step_iter = _streaming_batches(train_np, block_size, batch_size)
            state = None

        for step in step_iter:
            if train_regime == "random":
                x, y = _random_batch(train_np, batch_size, block_size)
                cur_b = int(x.shape[1])
            else:
                x, y, cur_b = step

            if hasattr(model, "init_state"):
                if (not stream_state) or state is None or (_state_batch_size(state) != cur_b):
                    state = _init_state_for_model(model, cur_b)
            else:
                state = None

            lr_step = scheduler.lr(global_step + 1)
            optimizer.learning_rate = lr_step

            if tbptt_len > 0 and tbptt_len < int(x.shape[0]) and hasattr(model, "init_state"):
                batch_loss = 0.0
                chunks = 0
                for start in range(0, int(x.shape[0]), tbptt_len):
                    end = min(int(x.shape[0]), start + tbptt_len)
                    x_chunk = x[start:end]
                    y_chunk = y[start:end]
                    (loss, state_out), grads = loss_and_grad(x_chunk, y_chunk, state)
                    if grad_clip > 0:
                        grads, _ = mxoptim.clip_grad_norm(grads, grad_clip)
                    optimizer.update(model, grads)
                    _apply_decoupled_weight_decay(model, wd_lookup, lr_step)
                    mx.eval(loss, model.parameters(), optimizer.state)
                    batch_loss += float(loss.item())
                    chunks += 1
                    state = _stop_grad_tree(state_out)
                epoch_loss += batch_loss / max(1, chunks)
                if not stream_state:
                    state = None
            else:
                (loss, state_out), grads = loss_and_grad(x, y, state)
                if grad_clip > 0:
                    grads, _ = mxoptim.clip_grad_norm(grads, grad_clip)
                optimizer.update(model, grads)
                _apply_decoupled_weight_decay(model, wd_lookup, lr_step)
                mx.eval(loss, model.parameters(), optimizer.state)
                epoch_loss += float(loss.item())
                if hasattr(model, "init_state") and stream_state:
                    state = _stop_grad_tree(state_out)
                else:
                    state = None

            global_step += 1
            _maybe_report_memory(global_step, mem_interval)
            _maybe_clear_memory(global_step, mem_clear_interval)

        val_ppl = _eval_perplexity(model, val_np, cfg)
        train_ce = epoch_loss / max(1, steps_per_epoch)
        _log_line(f"Epoch {epoch:02d}/{epochs} | Train CE: {train_ce:.4f} | Val PPL: {val_ppl:.2f}")

        save_checkpoint_entry(last_path, model, optimizer, {"global_step": global_step}, epoch, global_step, cfg, best_val)
        if val_ppl < best_val:
            best_val = val_ppl
            save_checkpoint_entry(best_path, model, optimizer, {"global_step": global_step}, epoch, global_step, cfg, best_val)

        if bool(_cfg_get(cfg, "save_each_epoch", False)):
            epoch_path = str(Path(save_dir) / f"epoch_{epoch:02d}_{run_tag}.pt")
            save_checkpoint_entry(epoch_path, model, optimizer, {"global_step": global_step}, epoch, global_step, cfg, best_val)

        _write_progress(
            run_dir,
            {
                "epoch": epoch,
                "train_ce": train_ce,
                "val_ppl": val_ppl,
                "best_val_ppl": best_val,
                "global_step": global_step,
            },
        )

    metrics: Dict[str, Optional[float]] = {"val_ppl": best_val}
    if test_np is not None:
        test_metrics = eval_metrics_entry(model, test_np, cfg)
        metrics["test_ppl"] = test_metrics["ppl"]
        metrics["gflops_per_token"] = test_metrics.get("gflops_per_token")
        metrics["act_sparsity"] = test_metrics.get("act_sparsity")
        _log_line(f"Test PPL: {test_metrics['ppl']:.2f}")
        _log_line("Measured GFLOPs/token (THOP): unavailable")
    return metrics
