#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CFG_LIST=(
  "${ROOT_DIR}/configs/wt2/neo_6m.yaml"
  "${ROOT_DIR}/configs/wt2/lstm_6m.yaml"
  "${ROOT_DIR}/configs/wt2/lstm_25m.yaml"
)

cd "$ROOT_DIR"

for CFG_PATH in "${CFG_LIST[@]}"; do
  if [[ ! -f "$CFG_PATH" ]]; then
    echo "Config not found: $CFG_PATH" >&2
    exit 1
  fi

  echo "== Training: ${CFG_PATH#${ROOT_DIR}/} ==" >&2
  if [[ -n "$BACKEND" ]]; then
    python3 -u scripts/train.py --config "$CFG_PATH" --device "$DEVICE" --backend "$BACKEND"
  else
    python3 -u scripts/train.py --config "$CFG_PATH" --device "$DEVICE"
  fi
done
