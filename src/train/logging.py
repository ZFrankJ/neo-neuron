"""Logging helpers."""

from typing import Optional

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def log_line(message: str) -> None:
    print(message, flush=True)


def mem_report(tag: str = "") -> None:
    if psutil is None:
        return
    rss = psutil.Process().memory_info().rss / (1024 ** 3)
    if torch.backends.mps.is_available():
        mps_alloc = torch.mps.current_allocated_memory() / (1024 ** 3)
        log_line(f"{tag} | RSS={rss:.2f} GB | MPS={mps_alloc:.2f} GB")
    else:
        log_line(f"{tag} | RSS={rss:.2f} GB")


def maybe_report_memory(step: int, interval: Optional[int]) -> None:
    if interval is None or interval <= 0:
        return
    if step % interval == 0:
        mem_report(tag=f"step {step}")
