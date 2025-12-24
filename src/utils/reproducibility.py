"""Reproducibility toggles and helpers."""

import os
from typing import Optional

import torch


def enable_determinism(seed: Optional[int] = None) -> None:
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def disable_determinism() -> None:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)
