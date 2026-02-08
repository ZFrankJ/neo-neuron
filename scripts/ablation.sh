#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
CFG_PATH="${ROOT_DIR}/configs/wt103/neo_20m.yaml"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Config not found: $CFG_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

python3 scripts/train.py --config "$CFG_PATH" --device "$DEVICE"
