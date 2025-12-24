"""Device selection utilities."""

from typing import Optional

import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
