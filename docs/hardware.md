# Hardware And Reproduction Checks

This repo separates local correctness checks from optional hardware validation.
MLX and PyTorch MPS checks can run on Apple Silicon. CUDA validation requires a
real Nvidia CUDA device.

## Standard Local Gate

Run this before trusting repo changes:

```bash
make check
```

This gate is intentionally CPU/skip-safe and does not prove CUDA parity.

## Apple Silicon Checks

On Apple Silicon, MLX and MPS checks are:

```bash
make mlx-parity
make mps-probe
```

`make mps-probe` is synthetic, no-checkpoint, opt-in diagnostics. It does not
validate checkpointed MPS, WT103-scale training, or production result rows.

## CUDA Preflight

Before making CUDA reproduction, parity, or performance claims, confirm the CUDA
probe runs on real Nvidia hardware:

```bash
python3 -m pytest -q tests/test_cuda_parity_harness.py
NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py
```

The first command validates the skip-safe contract. The second command is the
actual CUDA preflight. If it skips because CUDA is unavailable, the environment
has not validated CUDA parity.

The trusted CUDA baseline is:

- full precision
- `use_checkpoint: false`
- no `torch.compile`
- no fused optimizer
- no TF32 speed path

Standard GitHub-hosted runners for individual repositories are not treated as
Nvidia GPU runners for this project. CUDA CI should remain optional or
manual-only unless a real Nvidia runner is explicitly provisioned.
