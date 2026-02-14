#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

NORMS=(none rmsnorm layernorm)
BASE_CFGS=(
  "${ROOT_DIR}/configs/wt103/neo_40m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_40m.yaml"
)

run_with_norm() {
  local base_cfg="$1"
  local norm="$2"
  local tmp_cfg
  tmp_cfg="$(mktemp -t neo_stability.XXXXXX.yaml)"

  python3 - <<'PY' "$base_cfg" "$tmp_cfg" "$norm"
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])
norm = str(sys.argv[3]).strip().lower()

cfg = yaml.safe_load(base_cfg.read_text()) or {}
cfg["recurrent_norm"] = norm
cfg["resume_path"] = ""
cfg["run_tag"] = f"{cfg.get('run_tag', cfg.get('model_name', 'model'))}_{norm}"
tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  echo "== Training: ${base_cfg#${ROOT_DIR}/} | recurrent_norm=${norm} ==" >&2
  if [[ -n "$BACKEND" ]]; then
    python3 -u scripts/train.py --config "$tmp_cfg" --device "$DEVICE" --backend "$BACKEND"
  else
    python3 -u scripts/train.py --config "$tmp_cfg" --device "$DEVICE"
  fi
  rm -f "$tmp_cfg"
}

cd "$ROOT_DIR"

for base_cfg in "${BASE_CFGS[@]}"; do
  if [[ ! -f "$base_cfg" ]]; then
    echo "Config not found: $base_cfg" >&2
    exit 1
  fi
  for norm in "${NORMS[@]}"; do
    run_with_norm "$base_cfg" "$norm"
  done
done
