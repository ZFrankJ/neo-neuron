"""Seeding utilities."""

import os
import random
from typing import Optional

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
