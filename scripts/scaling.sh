#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-mlx}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_CFGS=(
  "${ROOT_DIR}/configs/wt103/neo_20m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_20m.yaml"
  "${ROOT_DIR}/configs/wt103/neo_30m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_30m.yaml"
  "${ROOT_DIR}/configs/wt103/neo_50m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_50m.yaml"
)

run_scaling_cfg() {
  local base_cfg="$1"
  local core_label=""
  local tmp_cfg
  tmp_cfg="$(mktemp -t neo_scaling.XXXXXX.yaml)"

  case "$base_cfg" in
    *neo_20m.yaml|*lstm_20m.yaml) core_label="~10M" ;;
    *neo_30m.yaml|*lstm_30m.yaml) core_label="~20M" ;;
    *neo_50m.yaml|*lstm_50m.yaml) core_label="~40M" ;;
    *) core_label="~unknown" ;;
  esac

  python3 - <<'PY' "$base_cfg" "$tmp_cfg"
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])

cfg = yaml.safe_load(base_cfg.read_text()) or {}
cfg["recurrent_norm"] = "rmsnorm"
cfg["recurrent_norm_place"] = "all"
cfg["resume_path"] = ""
cfg["run_tag"] = f"{cfg.get('run_tag', cfg.get('model_name', 'model'))}_scale_rmsnorm"
tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  echo "== Training: ${base_cfg#${ROOT_DIR}/} | core=${core_label} | recurrent_norm=rmsnorm ==" >&2
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
  run_scaling_cfg "$base_cfg"
done
