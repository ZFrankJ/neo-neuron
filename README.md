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
  and memory slope classification, and do not validate checkpointed MPS or large
  WT103 result production.
- PyTorch CUDA diagnostics are optional and must be run explicitly with
  `NEO_RUN_CUDA_PROBE=1 pytest tests/test_cuda_parity_harness.py` or
  `make cuda-probe` on a CUDA machine. The baseline is full precision,
  no checkpoint, no `torch.compile`, no fused optimizer, and no TF32 speed path.

## LSTM baseline controls

Existing LSTM configs retain the historical matched-control behavior: zero gate
biases, Xavier-uniform recurrent matrices, and `dropout` applied both between
recurrent layers and before the output projection. For a stronger, explicitly
labeled PyTorch LSTM baseline, configs can opt into:

```yaml
backend: torch
model_name: lstm
recurrent_norm: rmsnorm
forget_bias_init: 1.0
recurrent_init: orthogonal
lstm_layer_dropout: 0.0
dropout: 0.1
```

This path is the `standard-init RMSNorm-LSTM`. It uses a positive forget-gate
bias, gate-wise orthogonal hidden-to-hidden initialization, no LSTM-only
inter-layer dropout, and retains `dropout` before the language-model output.
The historical `RMSNorm-LSTM matched control` remains loadable and is not
silently reinterpreted. These initialization controls are PyTorch-only; MLX
runtime semantics remain unchanged.

## Config and result labels

Future WT103 Neo runs use `tanh`, and the checked-in Neo run tags include
`_tanh`. Historical `id4` and `id5` checkpoints and run snapshots retain their
original activation provenance; they are not tanh results. WT2 `6m` and `25m`
filenames remain as compatibility paths but are legacy labels, not exact
parameter counts. See `docs/training.md` for exact counts and paper-facing
small/large reporting labels.

## Transformer baseline

`configs/wt103/transformer_30m.yaml` is the opt-in `GPT-2-style Transformer
control`. It uses pre-norm causal blocks, GELU feed-forwards, learned absolute
positions, GPT-style 0.02 normal initialization with depth-scaled attention and
MLP residual projections, and the optimized causal attention primitive provided
by Torch or MLX. Torch and MLX checkpoint mapping is covered by a forward-parity
test.

Transformer configs without `transformer_variant` retain the historical
`legacy` implementation and checkpoint interpretation. Historical Transformer
runs are lightweight internal controls and must not be relabeled as GPT-2-style.

## Reproduction GPU preflight

Before making CUDA reproduction or performance claims, first verify that the
CUDA probe actually runs on an Nvidia GPU:

```bash
python3 -m pytest -q tests/test_cuda_parity_harness.py
NEO_RUN_CUDA_PROBE=1 python3 -m pytest -q tests/test_cuda_parity_harness.py
```

The first command should pass the skip-safe contract. The second command is the
real CUDA preflight: on a valid CUDA machine it should run the optional CUDA
single-step parity probe instead of skipping. If it skips with a CUDA unavailable
message, this environment has not validated CUDA parity and should not be used
for CUDA result claims. Standard GitHub-hosted runners for individual repos are
not treated as Nvidia GPU runners here; use a real local Nvidia machine or an
explicitly provisioned GPU runner.

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
make cuda-probe
```

`make mps-probe` is opt-in Apple Silicon diagnostics only. It does not validate checkpointed MPS, WT103 training, or production result quality. The strict backend parity PR queue lives in `docs/IMPLEMENTATION_PLAN.md`.
`make cuda-probe` is opt-in Nvidia CUDA diagnostics only. It skips without CUDA and does not make CUDA a required local or CI gate.

For macOS Apple Silicon MLX parity work, install dependencies with the pinned
parity constraints before running `make mlx-parity`:

```bash
python3 -m pip install -c requirements-lock/constraints-macos-mlx.txt -r requirements.txt
python3 -m pip install -c requirements-lock/constraints-macos-mlx.txt pytest
```

The constraints file pins the tested Torch/NumPy/MLX stack for macOS MLX parity.
Ubuntu CI continues to install plain `requirements.txt` and does not consume
macOS-only MLX constraints.

## Full evaluation via PyTorch

Full evaluation is still available through the PyTorch backend (including PPL, GFLOPs/token via THOP, and activation sparsity).

Use:

```bash
python3 scripts/eval.py --config configs/wt103/neo_20m.yaml --checkpoint checkpoints/best_wt103_neo_20m.pt --backend torch
python3 scripts/eval.py --config configs/wt103/lstm_20m.yaml --checkpoint checkpoints/best_wt103_lstm_20m.pt --backend torch
python3 scripts/eval.py --config configs/wt103/transformer_20m.yaml --checkpoint checkpoints/best_wt103_transformer_20m.pt --backend torch
```

Recurrent evaluation defaults to compatibility-preserving `block_reset`
semantics. Use `--eval-regime streaming` to carry state across contiguous
evaluation blocks. The command prints the selected regime; record it with any
reported perplexity. See `docs/training.md` for the exact contract.

For probing, use `--backend torch` as well.
