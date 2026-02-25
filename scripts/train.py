#!/usr/bin/env python3
"""Train a model from a config."""

import argparse
import sys
import warnings
from pathlib import Path

from _common import (
    build_data,
    build_model,
    count_params,
    ensure_repo_root_on_path,
    get_backend_api,
    infer_model_name,
    load_yaml,
    build_provenance,
    resolve_backend_name,
    resolve_run_dir,
    save_config_snapshot,
    save_metrics,
    validate_token_ids_against_vocab,
)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="multiprocessing.resource_tracker",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Train Neo models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--backend", default=None, help="torch | mlx (defaults to config/backend or torch)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--run-dir", default=None, help="Override run directory for artifacts")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from src.data import build_tokenizer

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    backend_name = resolve_backend_name(cfg, explicit=args.backend)
    backend = get_backend_api(backend_name)
    cfg["backend"] = backend_name

    if args.resume:
        cfg["resume_path"] = args.resume

    model_name = infer_model_name(cfg_path, cfg)
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    if "run_dir" not in cfg:
        cfg["run_dir"] = str(resolve_run_dir(cfg, model_name))
    run_dir = Path(cfg["run_dir"])
    save_config_snapshot(run_dir, cfg)
    print(f"Run directory: {run_dir}", flush=True)

    device = backend.get_runtime_device(args.device)
    seed = cfg.get("seed")
    if seed is not None:
        backend.seed_all(int(seed))

    model = build_model(cfg, model_name, backend_name=backend_name)

    param_count = count_params(model, backend_name=backend_name)
    print(f"Model: {model_name} | Params: {param_count/1e6:.1f}M", flush=True)

    tokenizer = build_tokenizer()
    train_ids, val_ids, test_ids = build_data(cfg, tokenizer)
    validate_token_ids_against_vocab(cfg, tokenizer, train_ids, val_ids, test_ids, context="train")
    metrics = backend.train_entry(model, cfg, train_ids, val_ids, test_ids=test_ids, device=device)

    if test_ids is not None and "test_ppl" in metrics:
        print(
            "Note: the in-training 'Test PPL' above is evaluated on the final model state "
            "(equivalent to last checkpoint), not the best checkpoint.",
            flush=True,
        )

    # Always run a PyTorch evaluation on the best checkpoint after training,
    # regardless of training backend (torch or mlx).
    save_dir = Path(str(cfg.get("save_dir", "checkpoints")))
    run_tag = str(cfg.get("run_tag", model_name))
    best_ckpt = save_dir / f"best_{run_tag}.pt"
    if test_ids is not None:
        if best_ckpt.exists():
            torch_backend = get_backend_api("torch")
            torch_device = torch_backend.get_runtime_device("cpu")
            torch_model = build_model(cfg, model_name, backend_name="torch")
            torch_backend.load_checkpoint_entry(str(best_ckpt), torch_model, device=torch_device)
            best_metrics = torch_backend.eval_metrics_entry(torch_model, test_ids, cfg, torch_device)
            metrics["best_ckpt_test_ppl_torch"] = best_metrics.get("ppl")
            metrics["best_ckpt_gflops_per_token_torch"] = best_metrics.get("gflops_per_token")
            metrics["best_ckpt_act_sparsity_torch"] = best_metrics.get("act_sparsity")
            print(f"Best checkpoint (PyTorch eval, CPU) Test PPL: {best_metrics['ppl']:.2f}", flush=True)
            if best_metrics.get("gflops_per_token") is not None:
                print(
                    f"Best checkpoint (PyTorch eval, CPU) GFLOPs/token: "
                    f"{best_metrics['gflops_per_token']:.3f}",
                    flush=True,
                )
            if best_metrics.get("act_sparsity") is not None:
                print(
                    f"Best checkpoint (PyTorch eval, CPU) Activation sparsity: "
                    f"{best_metrics['act_sparsity']:.4f}",
                    flush=True,
                )
        else:
            print(
                f"Warning: best checkpoint not found for post-training PyTorch eval: {best_ckpt}",
                file=sys.stderr,
                flush=True,
            )

    metrics["provenance"] = build_provenance(cfg, str(device), param_count, sys.argv, backend=backend_name)

    save_config_snapshot(run_dir, cfg)
    save_metrics(run_dir, metrics)
    print(f"Run artifacts saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
