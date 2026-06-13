"""Shared helpers for deterministic backend parity audit reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


ArrayLike = Any
NamedArrayMap = Mapping[str, ArrayLike]


@dataclass(frozen=True)
class ArrayDiffMetrics:
    """Finite max-diff plus non-finite counts for one array comparison."""

    max_abs_diff: float
    nan_count: int
    inf_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


@dataclass(frozen=True)
class DeviceAvailability:
    """Skip-safe optional hardware availability metadata."""

    device: str
    available: bool
    skip_reason: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "device": self.device,
            "available": self.available,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class BackendParityAuditReport:
    """Structured parity measurements shared by backend tests and probes."""

    backend_pair: tuple[str, str]
    device_pair: tuple[str, str]
    seed: int
    model_shape: dict[str, int | str | bool | float]
    use_checkpoint: bool
    loss_diff: float | None = None
    logits_max_diff: float | None = None
    state_max_diff: float | None = None
    gradient_max_diff: float | None = None
    update_max_diff: float | None = None
    grad_norm: float | None = None
    nan_count: int = 0
    inf_count: int = 0
    memory_samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_pair": list(self.backend_pair),
            "device_pair": list(self.device_pair),
            "seed": self.seed,
            "model_shape": dict(self.model_shape),
            "use_checkpoint": self.use_checkpoint,
            "loss_diff": self.loss_diff,
            "logits_max_diff": self.logits_max_diff,
            "state_max_diff": self.state_max_diff,
            "gradient_max_diff": self.gradient_max_diff,
            "update_max_diff": self.update_max_diff,
            "grad_norm": self.grad_norm,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "memory_samples": list(self.memory_samples),
        }


def _as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def collect_array_metrics(got: ArrayLike, expected: ArrayLike) -> ArrayDiffMetrics:
    """Collect finite max absolute diff and non-finite counts from both arrays."""

    got_array = _as_float_array(got)
    expected_array = _as_float_array(expected)
    if got_array.shape != expected_array.shape:
        raise ValueError(f"Shape mismatch: got {got_array.shape}, expected {expected_array.shape}")

    finite_mask = np.isfinite(got_array) & np.isfinite(expected_array)
    if np.any(finite_mask):
        max_abs_diff = float(np.max(np.abs(got_array[finite_mask] - expected_array[finite_mask])))
    else:
        max_abs_diff = 0.0
    return ArrayDiffMetrics(
        max_abs_diff=max_abs_diff,
        nan_count=int(np.isnan(got_array).sum() + np.isnan(expected_array).sum()),
        inf_count=int(np.isinf(got_array).sum() + np.isinf(expected_array).sum()),
    )


def collect_named_array_metrics(got: NamedArrayMap, expected: NamedArrayMap) -> ArrayDiffMetrics:
    got_keys = set(got)
    expected_keys = set(expected)
    if got_keys != expected_keys:
        missing = sorted(expected_keys - got_keys)
        extra = sorted(got_keys - expected_keys)
        raise ValueError(f"Named array keys differ: missing={missing}, extra={extra}")

    max_abs_diff = 0.0
    nan_count = 0
    inf_count = 0
    for name in sorted(got):
        metrics = collect_array_metrics(got[name], expected[name])
        max_abs_diff = max(max_abs_diff, metrics.max_abs_diff)
        nan_count += metrics.nan_count
        inf_count += metrics.inf_count
    return ArrayDiffMetrics(max_abs_diff=max_abs_diff, nan_count=nan_count, inf_count=inf_count)


def collect_named_array_l2_norm(values: NamedArrayMap) -> float:
    total = 0.0
    for value in values.values():
        array = _as_float_array(value)
        finite = array[np.isfinite(array)]
        total += float(np.sum(finite * finite))
    return float(np.sqrt(total))


def _combine_metrics(metrics: Sequence[ArrayDiffMetrics]) -> tuple[int, int]:
    return (
        sum(item.nan_count for item in metrics),
        sum(item.inf_count for item in metrics),
    )


def make_backend_parity_report(
    *,
    backend_pair: tuple[str, str],
    device_pair: tuple[str, str],
    seed: int,
    model_shape: Mapping[str, int | str | bool | float],
    use_checkpoint: bool,
    loss_pair: tuple[float, float] | None = None,
    logits_pair: tuple[ArrayLike, ArrayLike] | None = None,
    state_pair: tuple[ArrayLike, ArrayLike] | None = None,
    gradient_pair: tuple[NamedArrayMap, NamedArrayMap] | None = None,
    update_pair: tuple[NamedArrayMap, NamedArrayMap] | None = None,
    memory_samples: list[dict[str, Any]] | None = None,
) -> BackendParityAuditReport:
    metrics: list[ArrayDiffMetrics] = []

    loss_diff = None
    if loss_pair is not None:
        loss_diff = abs(float(loss_pair[0]) - float(loss_pair[1]))

    logits_max_diff = None
    if logits_pair is not None:
        logits_metrics = collect_array_metrics(*logits_pair)
        metrics.append(logits_metrics)
        logits_max_diff = logits_metrics.max_abs_diff

    state_max_diff = None
    if state_pair is not None:
        state_metrics = collect_array_metrics(*state_pair)
        metrics.append(state_metrics)
        state_max_diff = state_metrics.max_abs_diff

    gradient_max_diff = None
    grad_norm = None
    if gradient_pair is not None:
        gradient_metrics = collect_named_array_metrics(*gradient_pair)
        metrics.append(gradient_metrics)
        gradient_max_diff = gradient_metrics.max_abs_diff
        grad_norm = collect_named_array_l2_norm(gradient_pair[0])

    update_max_diff = None
    if update_pair is not None:
        update_metrics = collect_named_array_metrics(*update_pair)
        metrics.append(update_metrics)
        update_max_diff = update_metrics.max_abs_diff

    nan_count, inf_count = _combine_metrics(metrics)
    return BackendParityAuditReport(
        backend_pair=backend_pair,
        device_pair=device_pair,
        seed=int(seed),
        model_shape=dict(model_shape),
        use_checkpoint=bool(use_checkpoint),
        loss_diff=loss_diff,
        logits_max_diff=logits_max_diff,
        state_max_diff=state_max_diff,
        gradient_max_diff=gradient_max_diff,
        update_max_diff=update_max_diff,
        grad_norm=grad_norm,
        nan_count=nan_count,
        inf_count=inf_count,
        memory_samples=[] if memory_samples is None else list(memory_samples),
    )


def device_availability(device: str) -> DeviceAvailability:
    """Return availability metadata without forcing optional hardware tests to run."""

    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return DeviceAvailability(device="cpu", available=True)
    if normalized not in {"mps", "cuda"}:
        return DeviceAvailability(device=normalized, available=False, skip_reason=f"unsupported device '{normalized}'")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - only hit if torch import itself is broken.
        return DeviceAvailability(device=normalized, available=False, skip_reason=f"PyTorch unavailable: {exc}")

    if normalized == "cuda":
        available = bool(torch.cuda.is_available())
        reason = "" if available else "PyTorch CUDA backend is not available"
        return DeviceAvailability(device="cuda", available=available, skip_reason=reason)

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    reason = "" if has_mps else "PyTorch MPS backend is not available"
    return DeviceAvailability(device="mps", available=bool(has_mps), skip_reason=reason)
