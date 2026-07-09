#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neo_mplconfig}"

cd "$ROOT_DIR"

run() {
  printf '\n== %s ==\n' "$*"
  "$@"
}

printf '== Environment ==\n'
"$PYTHON" - <<'PY'
import platform
import sys

import torch

print("python", sys.version.split()[0])
print("platform", platform.platform())
print("torch", torch.__version__)
print("mps_built", hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
print("mps_available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
print("cuda_available", torch.cuda.is_available())
PY

run "$PYTHON" -m pytest -q \
  tests/test_backend_parity_audit.py \
  tests/test_checkpoint_metadata.py \
  tests/test_checkpoint_resume.py \
  tests/test_forward_shapes.py \
  tests/test_gain_tanh_activation.py \
  tests/test_imports.py \
  tests/test_probe_hooks.py \
  tests/test_smoke_train_cpu.py \
  tests/test_cuda_parity_harness.py

printf '\n== MLX preflight ==\n'
if "$PYTHON" - <<'PY'
import subprocess
import sys

code = """
import mlx.core as mx
import mlx.nn as mxnn
import mlx.optimizers as mxoptim
print("mlx.core import ok")
print("default_device", mx.default_device())
"""
proc = subprocess.run(
    [sys.executable, "-c", code],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if proc.stdout.strip():
    print(proc.stdout.strip())
if proc.returncode != 0:
    print(f"MLX import failed with exit code {proc.returncode}")
    stderr_lines = [line for line in proc.stderr.splitlines() if line.strip()]
    for line in stderr_lines[:8]:
        print(line)
    raise SystemExit(proc.returncode)
PY
then
  run "$PYTHON" -m pytest -q tests/test_mlx_reference_parity.py
else
  printf 'MLX preflight failed; skipped tests/test_mlx_reference_parity.py.\n'
  printf 'Use requirements-lock/constraints-macos-mlx.txt in a clean macOS arm64 environment before making MLX parity claims.\n'
fi

run env NEO_RUN_MPS_PROBE=1 "$PYTHON" -m pytest -q -rs tests/test_mps_no_checkpoint_probe.py

printf '\n== Summary ==\n'
printf 'Torch CPU path passed if the first pytest block passed.\n'
printf 'MPS is validated only if the MPS probe ran without skips.\n'
printf 'MLX parity is validated only if the MLX preflight and MLX parity test block passed.\n'
