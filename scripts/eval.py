#!/usr/bin/env python3
"""Evaluate a checkpoint."""

import argparse
from pathlib import Path

from _common import (
    build_data,
    build_model,
    ensure_repo_root_on_path,
    get_backend_api,
    infer_model_name,
    load_yaml,
    resolve_backend_name,
    validate_token_ids_against_vocab,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Neo models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--backend", default=None, help="torch | mlx (defaults to config/backend or torch)")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from src.data import build_tokenizer

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    backend_name = resolve_backend_name(cfg, explicit=args.backend)
    backend = get_backend_api(backend_name)
    cfg["backend"] = backend_name

    device = backend.get_runtime_device(args.device)
    seed = cfg.get("seed")
    if seed is not None:
        backend.seed_all(int(seed))

    tokenizer = build_tokenizer()
    train_ids, val_ids, test_ids = build_data(cfg, tokenizer)
    validate_token_ids_against_vocab(cfg, tokenizer, train_ids, val_ids, test_ids, context="eval")

    model_name = infer_model_name(cfg_path, cfg)
    model = build_model(cfg, model_name, backend_name=backend_name)
    backend.load_checkpoint_entry(args.checkpoint, model, device=device)

    ids = val_ids if args.split == "val" else test_ids
    metrics = backend.eval_metrics_entry(model, ids, cfg, device)
    print(f"{args.split} PPL: {metrics['ppl']:.2f}")
    if metrics.get("gflops_per_token") is not None:
        print(f"Measured GFLOPs/token (THOP): {metrics['gflops_per_token']:.3f}")
    else:
        print("Measured GFLOPs/token (THOP): unavailable")
    if metrics.get("act_sparsity") is not None:
        print(f"Activation sparsity: {metrics['act_sparsity']:.4f}")


if __name__ == "__main__":
    main()
