"""Backend loader utilities."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List


_BACKEND_MODULES: Dict[str, str] = {
    "torch": "src.runtime.backends.torch_backend",
    "mlx": "src.runtime.backends.mlx_backend",
}


def available_backends() -> List[str]:
    return sorted(_BACKEND_MODULES)


def resolve_backend_name(cfg: Dict[str, Any], explicit: str | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        name = str(explicit).strip().lower()
    else:
        name = str(cfg.get("backend", "torch")).strip().lower()
    if name not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported backend '{name}'. Available: {available_backends()}")
    return name


def get_backend(name: str):
    key = str(name).strip().lower()
    if key not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported backend '{key}'. Available: {available_backends()}")
    module_path = _BACKEND_MODULES[key]
    return importlib.import_module(module_path)
