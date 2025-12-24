#!/usr/bin/env python3
"""Train a model from a config."""

import argparse
import os
import sys
from pathlib import Path

from _common import (
    build_data,
    build_model,
    count_params,
    ensure_repo_root_on_path,
    infer_model_name,
    load_yaml,
    build_provenance,
    resolve_run_dir,
    save_config_snapshot,
    save_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neo models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from src.data import build_tokenizer
    from src.train import train_model
    from src.train.trainer import RestartEpoch
    from src.utils import get_device, set_seed

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    if args.resume:
        cfg["resume_path"] = args.resume

    device = get_device(args.device) if args.device and args.device != "auto" else get_device()
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    tokenizer = build_tokenizer()
    train_ids, val_ids, test_ids = build_data(cfg, tokenizer)

    model_name = infer_model_name(cfg_path, cfg)
    model = build_model(cfg, model_name)

    param_count = count_params(model)
    print(f"Model: {model_name} | Params: {param_count/1e6:.2f}M", flush=True)
    try:
        metrics = train_model(model, cfg, train_ids, val_ids, test_ids=test_ids, device=device)
    except RestartEpoch as exc:
        clean_args = []
        skip_next = False
        for arg in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if arg == "--resume":
                skip_next = True
                continue
            clean_args.append(arg)
        restart_args = [sys.executable, sys.argv[0]] + clean_args + ["--resume", exc.resume_path]
        print(f"[Restart] Relaunching: {' '.join(restart_args)}", flush=True)
        os.execv(sys.executable, restart_args)
    metrics["provenance"] = build_provenance(cfg, str(device), param_count, sys.argv)

    run_dir = resolve_run_dir(cfg, model_name)
    cfg["run_dir"] = str(run_dir)
    save_config_snapshot(run_dir, cfg)
    save_metrics(run_dir, metrics)
    print(f"Run artifacts saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
