#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-mlx}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

cd "$ROOT_DIR"

echo "== Stage 1: Ablation (Neo/LSTM 20M, none/rmsnorm/layernorm) ==" >&2
./scripts/ablation.sh "$DEVICE" "$BACKEND"

echo "== Stage 2: Scaling (Neo/LSTM 20/40/80M, RMSNorm) ==" >&2
./scripts/scaling.sh "$DEVICE" "$BACKEND"

echo "== Stage 3: Transformer 40M training ==" >&2
if [[ -n "$BACKEND" ]]; then
  python3 -u scripts/train.py --config configs/wt103/transformer_40m.yaml --device "$DEVICE" --backend "$BACKEND"
else
  python3 -u scripts/train.py --config configs/wt103/transformer_40m.yaml --device "$DEVICE"
fi

echo "== Stage 4: Variance (Neo/LSTM 20M, no norm, 5 seeds) ==" >&2
./scripts/variance.sh "$DEVICE" "$BACKEND"

python3 scripts/summarize_runs.py --root runs --out experiments/overall_summary
