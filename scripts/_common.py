"""Shared helpers for scripts."""

from __future__ import annotations

import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict


def ensure_repo_root_on_path() -> Path:
    # Temporary until package installation or module execution is set up.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_backend_name(cfg: Dict[str, Any], explicit: str | None = None) -> str:
    ensure_repo_root_on_path()
    from src.runtime import resolve_backend_name as _resolve_backend_name

    return _resolve_backend_name(cfg, explicit=explicit)


def get_backend_api(backend_name: str):
    ensure_repo_root_on_path()
    from src.runtime import get_backend

    return get_backend(backend_name)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - configuration guard
        raise RuntimeError(
            "This script requires 'pyyaml'.\n"
            "Install via: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return data


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - configuration guard
        raise RuntimeError(
            "This script requires 'pyyaml'.\n"
            "Install via: pip install pyyaml"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def infer_model_name(config_path: Path, cfg: Dict[str, Any]) -> str:
    if "model_name" in cfg:
        return str(cfg["model_name"]).lower().strip()
    print(
        f"Warning: 'model_name' missing in {config_path}. Falling back to heuristic.",
        file=sys.stderr,
        flush=True,
    )
    if "n_heads" in cfg and "ff_mult" in cfg:
        return "transformer"
    if "cell_type" in cfg:
        return "neo"
    return "lstm"


def build_model(cfg: Dict[str, Any], model_name: str, backend_name: str = "torch"):
    backend = get_backend_api(backend_name)
    return backend.build_model(cfg, model_name)


def build_data(cfg: Dict[str, Any], tokenizer):
    ensure_repo_root_on_path()
    from src.data import load_autoregressive_corpus

    dataset_name = str(cfg.get("dataset_name", "wikitext"))
    dataset_config = str(cfg.get("dataset_config", "wikitext-2-raw-v1"))
    train_split = str(cfg.get("train_split", "train"))
    val_split = str(cfg.get("val_split", "validation"))
    test_split = str(cfg.get("test_split", "test"))
    cache_root = cfg.get("data_root")
    if cache_root is not None:
        cache_root = Path(cache_root)

    return load_autoregressive_corpus(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        tokenizer=tokenizer,
        cache_root=cache_root,
    )


def count_params(model, backend_name: str = "torch") -> int:
    backend = get_backend_api(backend_name)
    return int(backend.count_params(model))


def resolve_run_dir(cfg: Dict[str, Any], model_name: str) -> Path:
    runs_root = Path(cfg.get("runs_dir", "runs"))
    run_tag = cfg.get("run_tag", model_name)
    dataset_config = cfg.get("dataset_config", "dataset")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / str(dataset_config) / model_name / str(run_tag) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: Path, cfg: Dict[str, Any]) -> None:
    save_yaml(run_dir / "config.yaml", cfg)


def save_metrics(run_dir: Path, metrics: Dict[str, Any]) -> None:
    save_json(run_dir / "metrics.json", metrics)


def get_git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_provenance(
    cfg: Dict[str, Any],
    device: str,
    param_count: int,
    argv: list,
    backend: str = "torch",
) -> Dict[str, Any]:
    root = ensure_repo_root_on_path()
    return {
        "git_commit": get_git_commit(root),
        "command": " ".join(argv),
        "device": device,
        "backend": backend,
        "param_count": param_count,
        "seed": cfg.get("seed"),
    }


def validate_token_ids_against_vocab(cfg: Dict[str, Any], tokenizer, *splits, context: str = "run") -> None:
    vocab_size = int(cfg.get("vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocab_size={vocab_size}. Set a positive vocab_size in config.")

    max_id = -1
    for split in splits:
        if split is None:
            continue
        if hasattr(split, "numel") and split.numel() == 0:  # torch tensor
            continue
        if hasattr(split, "size") and not callable(split.size) and split.size == 0:  # numpy array
            continue
        if hasattr(split, "max"):
            cur = int(split.max().item() if hasattr(split.max(), "item") else split.max())
            if cur > max_id:
                max_id = cur

    if max_id >= vocab_size:
        tok_name = getattr(tokenizer, "name_or_path", "tokenizer")
        tok_vocab = getattr(tokenizer, "vocab_size", None)
        hint = f"Tokenizer '{tok_name}' vocab_size={tok_vocab}" if tok_vocab is not None else f"Tokenizer '{tok_name}'"
        raise ValueError(
            f"[{context}] token id range exceeds config vocab_size: max_token_id={max_id}, "
            f"requires vocab_size >= {max_id + 1}, but config has vocab_size={vocab_size}. "
            f"{hint}. This can cause embedding OOB and backend crashes on Metal/MLX."
        )
