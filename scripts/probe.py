#!/usr/bin/env python3
"""Probe neuron activations and gates."""

import argparse
import random
from pathlib import Path

import torch

from _common import (
    build_data,
    build_model,
    ensure_repo_root_on_path,
    get_backend_api,
    infer_model_name,
    load_yaml,
    resolve_backend_name,
    save_json,
    validate_token_ids_against_vocab,
)


def _make_probe_batch(ids: torch.Tensor, seq_len: int, batch_size: int, start: int) -> torch.Tensor:
    n_tokens = ids.size(0)
    max_start = max(0, n_tokens - (seq_len + 1))
    start = min(start, max_start)
    if batch_size <= 1:
        idx = ids[start : start + seq_len].unsqueeze(1)
        return idx

    offsets = [start + i * seq_len for i in range(batch_size)]
    idx_list = []
    for off in offsets:
        off = min(off, max_start)
        idx_list.append(ids[off : off + seq_len])
    return torch.stack(idx_list, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Neo/LSTM activations")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers to probe")
    parser.add_argument("--neurons-per-layer", type=int, default=3, help="Neurons per layer")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--start", type=int, default=0, help="Start offset in validation set")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--backend", default=None, help="torch | mlx (defaults to config/backend or torch)")
    parser.add_argument("--save-plots", action="store_true", help="Save plots without displaying them")
    args = parser.parse_args()

    ensure_repo_root_on_path()
    from src.data import build_tokenizer
    from src.probe import (
        capture_traces,
        plot_lstm_records,
        plot_neo_records,
        select_random_neurons,
        summarize_neuron_records,
    )
    from src.runtime.checkpoint_compat import load_checkpoint_payload
    from src.utils import set_seed

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    backend_name = resolve_backend_name(cfg, explicit=args.backend)
    backend = get_backend_api(backend_name)
    cfg["backend"] = backend_name

    if not backend.supports_probe():
        raise ValueError(
            f"Probe is currently implemented for torch backend only. Got backend='{backend_name}'."
        )

    device = backend.get_runtime_device(args.device)
    set_seed(int(args.seed))

    ckpt = load_checkpoint_payload(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("cfg")
    if isinstance(ckpt_cfg, dict) and ckpt_cfg:
        merged_cfg = dict(cfg)
        merged_cfg.update(ckpt_cfg)
        cfg = merged_cfg

    tokenizer = build_tokenizer()
    train_ids, val_ids, test_ids = build_data(cfg, tokenizer)
    validate_token_ids_against_vocab(cfg, tokenizer, train_ids, val_ids, test_ids, context="probe")

    model_name = infer_model_name(cfg_path, cfg)
    if model_name == "transformer":
        raise ValueError("Transformer probing is not implemented.")

    model = build_model(cfg, model_name, backend_name=backend_name)
    backend.load_checkpoint_entry(args.checkpoint, model, device=device)
    model.to(device)

    idx = _make_probe_batch(val_ids, args.seq_len, args.batch_size, args.start).to(device)
    layer_indices, selected_neurons = select_random_neurons(
        model,
        n_layers_to_watch=args.layers,
        neurons_per_layer=args.neurons_per_layer,
        seed=args.seed,
    )

    records = capture_traces(model, idx, layer_indices, selected_neurons)
    summary = summarize_neuron_records(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_dir / "records.pt")
    save_json(out_dir / "summary.json", summary)

    global_stats = summary.get("_global_abs_percentiles")
    if isinstance(global_stats, dict):
        def _fmt_abs_stats(name: str) -> None:
            stats = global_stats.get(name)
            if not isinstance(stats, dict):
                return
            print(
                f"{name} |abs| "
                f"P50={float(stats.get('p50', 0.0)):.6f} "
                f"P90={float(stats.get('p90', 0.0)):.6f} "
                f"P95={float(stats.get('p95', 0.0)):.6f} "
                f"P99={float(stats.get('p99', 0.0)):.6f} "
                f"Max={float(stats.get('max', 0.0)):.6f}",
                flush=True,
            )

        for key in ("state", "output", "hidden_state", "cell_state", "f_x_raw", "g_x_raw"):
            _fmt_abs_stats(key)

    if args.save_plots:
        if model_name == "neo":
            if plot_neo_records is None:
                raise RuntimeError("Plotting requires matplotlib. Install via: pip install matplotlib")
            plot_neo_records(records, args.seq_len, str(out_dir))
        else:
            if plot_lstm_records is None:
                raise RuntimeError("Plotting requires matplotlib. Install via: pip install matplotlib")
            plot_lstm_records(records, args.seq_len, str(out_dir))


if __name__ == "__main__":
    main()
