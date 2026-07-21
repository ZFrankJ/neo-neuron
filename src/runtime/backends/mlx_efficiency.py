"""MLX execution, synchronization, and telemetry benchmark adapter."""

from __future__ import annotations

import math
import platform
from typing import Any, Mapping

import numpy as np
import psutil

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as mxnn
    import mlx.optimizers as mxoptim
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("MLX benchmark requested but 'mlx' is not installed") from exc

from ..efficiency import BenchmarkObservation
from .mlx_backend import (
    _apply_decoupled_weight_decay,
    _build_weight_decay_lookup,
    _to_betas,
)


class MlxEfficiencyAdapter:
    backend = "mlx"
    framework_version = str(getattr(mlx, "__version__", "unknown"))

    def __init__(self, model, cfg: Mapping[str, Any], device: Any):
        self.model = model
        self.cfg = dict(cfg)
        self.device = str(device)
        if self.device not in {"cpu", "gpu"}:
            raise ValueError(f"Unsupported MLX benchmark device '{self.device}'")
        parameters = list(tree_flatten(model.parameters()))
        self.dtype = str(parameters[0][1].dtype) if parameters else "unknown"
        self.hardware_identifier = f"{platform.machine()} | {platform.processor() or 'Apple Silicon'}"
        self.synchronization_policy = "mx.eval(pending); mx.synchronize()"
        self.telemetry_capabilities = {
            "process_rss": True,
            "backend_active_memory": hasattr(mx, "get_active_memory"),
            "backend_peak_memory": hasattr(mx, "get_peak_memory"),
        }
        self.tokens = None
        self.targets = None
        self.optimizer = None
        self.loss_and_grad = None
        self.weight_decay_lookup = None
        self.workload = None
        self._pending_eval = []

    def parameter_entries(self):
        return [
            (name, int(np.prod(value.shape)), True)
            for name, value in tree_flatten(self.model.parameters())
        ]

    def prepare(self, tokens: np.ndarray, targets: np.ndarray) -> None:
        self.tokens = mx.array(tokens, dtype=mx.int32)
        self.targets = mx.array(targets, dtype=mx.int32)

    def configure(self, workload: str) -> None:
        self.workload = workload
        if workload == "train_step":
            self.model.train()
            self.optimizer = mxoptim.AdamW(
                learning_rate=float(self.cfg.get("lr", 3e-4)),
                betas=list(_to_betas(self.cfg.get("betas"), (0.9, 0.95))),
                eps=float(self.cfg.get("adam_eps", 1e-8)),
                weight_decay=0.0,
            )
            model_name = str(self.cfg.get("model_name", "unknown")).strip().lower()
            self.weight_decay_lookup = _build_weight_decay_lookup(
                self.model,
                model_name,
                self.cfg,
            )

            def loss_fn(tokens, targets):
                logits, _ = self.model(tokens, None)
                return mxnn.losses.cross_entropy(logits, targets, reduction="mean"), logits

            self.loss_and_grad = mxnn.value_and_grad(self.model, loss_fn)
        else:
            self.model.eval()

    def reset_peak_memory(self) -> None:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

    def synchronize(self) -> None:
        if self._pending_eval:
            mx.eval(*self._pending_eval)
            self._pending_eval = []
        mx.synchronize()

    def _forward_sequence(self):
        return self.model(self.tokens, None)[0]

    def _forward_streaming(self):
        state = None
        outputs = []
        if hasattr(self.model, "init_state"):
            for index in range(int(self.tokens.shape[0])):
                logits, state = self.model(self.tokens[index : index + 1], state)
                outputs.append(logits)
        else:
            for index in range(int(self.tokens.shape[0])):
                logits, _ = self.model(self.tokens[: index + 1], None)
                outputs.append(logits[-1:])
        return mx.concatenate(outputs, axis=0)

    def run(self, workload: str):
        if workload != self.workload:
            raise ValueError(f"Adapter configured for {self.workload!r}, got {workload!r}")
        if workload == "train_step":
            assert self.optimizer is not None
            assert self.loss_and_grad is not None
            assert self.weight_decay_lookup is not None
            (loss, logits), grads = self.loss_and_grad(self.tokens, self.targets)
            grad_clip = float(self.cfg.get("grad_clip", 1.0))
            if grad_clip > 0:
                grads, _ = mxoptim.clip_grad_norm(grads, grad_clip)
            self.optimizer.update(self.model, grads)
            _apply_decoupled_weight_decay(
                self.model,
                self.weight_decay_lookup,
                float(self.optimizer.learning_rate),
            )
            self._pending_eval = [loss, logits, self.model.parameters(), self.optimizer.state]
            return {"logits": logits, "scalar": loss}
        logits = self._forward_streaming() if workload == "streaming_decode" else self._forward_sequence()
        scalar = mx.mean(logits.astype(mx.float32))
        self._pending_eval = [logits, scalar]
        return {"logits": logits, "scalar": scalar}

    def finalize_observation(self, pending) -> BenchmarkObservation:
        logits = np.asarray(pending["logits"])
        scalar = float(np.asarray(pending["scalar"]).item())
        return BenchmarkObservation(
            output_shape=tuple(int(value) for value in logits.shape),
            finite=bool(np.isfinite(logits).all() and math.isfinite(scalar)),
            scalar=scalar,
        )

    def memory_snapshot(self):
        active = int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else None
        peak = int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
        return {
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "backend_active_bytes": active,
            "backend_peak_bytes": peak,
        }
