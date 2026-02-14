#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SEEDS=(42 314 1024 61803 271828)
BASE_CFGS=(
  "${ROOT_DIR}/configs/wt103/neo_20m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_20m.yaml"
)

run_with_seed() {
  local base_cfg="$1"
  local seed="$2"
  local tmp_cfg
  tmp_cfg="$(mktemp -t neo_variance.XXXXXX.yaml)"

  python3 - <<'PY' "$base_cfg" "$tmp_cfg" "$seed"
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = yaml.safe_load(base_cfg.read_text()) or {}
cfg["recurrent_norm"] = "none"
cfg["seed"] = seed
cfg["resume_path"] = ""
cfg["run_tag"] = f"{cfg.get('run_tag', cfg.get('model_name', 'model'))}_nonorm_seed{seed}"
tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  echo "== Training: ${base_cfg#${ROOT_DIR}/} | seed=${seed} | recurrent_norm=none ==" >&2
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
  for seed in "${SEEDS[@]}"; do
    run_with_seed "$base_cfg" "$seed"
  done
done
