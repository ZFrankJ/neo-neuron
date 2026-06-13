> **Warning**:
> 1. The codes are only verified on Torch MPS and MLX, so check compatibility first if you are using CUDA.
> 2. This repo's code is mainly generated through vibe coding (>=90%), so you should better recheck.
> 3. This is my first time to create GitHub repo for scientific researches, may have mistakes.
> - *If any of the above causes problems and errors in reproducibility to you, I will be very sorry for wasting your time and please let me know, I will fix as soon as I can.*

# Neo Neuron: a new parameter-efficient recurrent cell

This repo is for demonstrating a new recurrent neuron founded by me. 

# This repo is currently still in progress, please wait for training completed and results handled.


## Backend selection

This repo now supports two execution backends behind the same script interface:

- `torch` (default, full feature path)
- `mlx` (Apple Silicon / MLX path for training and eval)

You can choose backend in either config or CLI:

```yaml
backend: torch
```

```bash
python3 scripts/train.py --config configs/wt103/neo_20m.yaml --backend mlx
python3 scripts/eval.py --config configs/wt103/neo_20m.yaml --checkpoint checkpoints/best_wt103_neo_20m.pt --backend mlx
```

## Current limitations

- `probe.py` is currently implemented for `torch` backend only.
- `mlx` backend evaluation currently reports perplexity, while GFLOPs/token and activation sparsity are marked unavailable.
- Checkpoints are stored in a unified model-state format and can be loaded across `torch` and `mlx` for model weights.
- Optimizer/scheduler state remains backend-native and is restored only when backend matches.
- Neo checkpoints now carry alignment metadata such as `reference_backend`, `rmsnorm_eps`,
  `activation_id`, recurrent norm settings, `use_checkpoint`, and `weight_decay_policy`.
  Missing metadata is treated as legacy/provisional, while mismatches with the requested
  eval/resume config fail before evaluation.
- PyTorch MPS diagnostics are optional and must be run explicitly with
  `NEO_RUN_MPS_PROBE=1 pytest tests/test_mps_no_checkpoint_probe.py`. They cover
  only tiny synthetic no-checkpoint single-step parity, short trajectory parity,
  and memory probes, and do not validate checkpointed MPS or large WT103 result
  production.

## Development workflow

This repo uses the local `codex-harness` workflow adapted for Neo.

Standard local checks:

```bash
make check
```

Useful focused checks:

```bash
make test
make mlx-parity
make mps-probe
```

`make mps-probe` is opt-in Apple Silicon diagnostics only. It does not validate checkpointed MPS, WT103 training, or production result quality. The strict backend parity PR queue lives in `docs/IMPLEMENTATION_PLAN.md`.

## Full evaluation via PyTorch

Full evaluation is still available through the PyTorch backend (including PPL, GFLOPs/token via THOP, and activation sparsity).

Use:

```bash
python3 scripts/eval.py --config configs/wt103/neo_20m.yaml --checkpoint checkpoints/best_wt103_neo_20m.pt --backend torch
python3 scripts/eval.py --config configs/wt103/lstm_20m.yaml --checkpoint checkpoints/best_wt103_lstm_20m.pt --backend torch
python3 scripts/eval.py --config configs/wt103/transformer_20m.yaml --checkpoint checkpoints/best_wt103_transformer_20m.pt --backend torch
```

For probing, use `--backend torch` as well.
