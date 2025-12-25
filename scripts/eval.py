#!/usr/bin/env python3
"""Evaluate a checkpoint."""

import argparse
from pathlib import Path

from _common import build_data, build_model, ensure_repo_root_on_path, infer_model_name, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Neo models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from src.data import build_tokenizer
    from src.train.eval import evaluate_metrics
    from src.train import load_checkpoint
    from src.utils import get_device, set_seed

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    device = get_device(args.device) if args.device and args.device != "auto" else get_device()
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    tokenizer = build_tokenizer()
    train_ids, val_ids, test_ids = build_data(cfg, tokenizer)

    model_name = infer_model_name(cfg_path, cfg)
    model = build_model(cfg, model_name)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    ids = val_ids if args.split == "val" else test_ids
    metrics = evaluate_metrics(model, ids, cfg, device)
    print(f"{args.split} PPL: {metrics['ppl']:.2f}")
    if metrics.get("gflops_per_token") is not None:
        print(f"Measured GFLOPs/token (THOP): {metrics['gflops_per_token']:.3f}")
    else:
        print("Measured GFLOPs/token (THOP): unavailable")
    if metrics.get("act_sparsity") is not None:
        print(f"Activation sparsity: {metrics['act_sparsity']:.4f}")


if __name__ == "__main__":
    main()
