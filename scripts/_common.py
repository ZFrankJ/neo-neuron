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


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Missing required config key: '{key}'")
    return cfg[key]


def build_model(cfg: Dict[str, Any], model_name: str):
    ensure_repo_root_on_path()
    from src.models import LSTMLM, NeoLM, TransformerLM

    vocab_size = int(_require(cfg, "vocab_size"))
    d_model = int(_require(cfg, "d_model"))
    d_embed = int(cfg.get("d_embed", d_model))
    n_layers = int(cfg.get("n_layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    tie_embeddings = bool(cfg.get("tie_embeddings", True))

    if model_name == "lstm":
        return LSTMLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
        )

    if model_name == "neo":
        cell_kwargs = {
            "alpha_init": float(cfg.get("alpha_init", 1e-1)),
            "alpha_trainable": bool(cfg.get("alpha_trainable", True)),
            "alpha_min": float(cfg.get("alpha_min", 1e-2)),
            "alpha_max": float(cfg.get("alpha_max", 1e0)),
            "rms_norm_fx": bool(cfg.get("rms_norm_fx", True)),
            "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-5)),
            "g_clamp_L": float(cfg.get("g_clamp_L", 1.0)),
        }
        return NeoLM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_embed,
            n_layers=n_layers,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
            cell_type=str(cfg.get("cell_type", "cortical")),
            cell_kwargs=cell_kwargs,
            use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        )

    if model_name == "transformer":
        n_heads = int(_require(cfg, "n_heads"))
        return TransformerLM(
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


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def build_provenance(cfg: Dict[str, Any], device: str, param_count: int, argv: list) -> Dict[str, Any]:
    root = ensure_repo_root_on_path()
    return {
        "git_commit": get_git_commit(root),
        "command": " ".join(argv),
        "device": device,
        "param_count": param_count,
        "seed": cfg.get("seed"),
    }
