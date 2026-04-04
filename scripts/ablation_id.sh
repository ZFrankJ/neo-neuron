#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-mlx}"
ACTIVATIONS_CSV="${3:-id3,id4,id5,tanh,gelu,none}"
BASE_CFG="${ROOT_DIR}/configs/wt103/neo_20m.yaml"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

IFS=',' read -r -a ACTIVATIONS <<<"$ACTIVATIONS_CSV"
if [[ "${#ACTIVATIONS[@]}" -eq 0 ]]; then
  echo "No activations provided. Pass a CSV list like: id3,id4,id5,tanh,gelu,none" >&2
  exit 1
fi

run_with_activation() {
  local activation_name="$1"
  local tmp_cfg
  tmp_cfg="$(mktemp -t neo_ablation_id.XXXXXX.yaml)"

  python3 - <<'PY' "$BASE_CFG" "$tmp_cfg" "$activation_name"
import re
import sys
from pathlib import Path
import yaml

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])
activation_name = str(sys.argv[3]).strip().lower()

allowed = {"id3", "id4", "id5", "tanh", "gelu", "none", "identity"}
if activation_name not in allowed:
    raise ValueError(
        f"Unsupported activation '{activation_name}'. Expected one of: "
        "id3,id4,id5,tanh,gelu,none."
    )

cfg = yaml.safe_load(base_cfg.read_text()) or {}
cfg["resume_path"] = ""
cfg["epochs"] = 5
cfg["activation_id"] = activation_name
suffix = re.sub(r"[^a-z0-9]+", "_", activation_name).strip("_")
cfg["run_tag"] = f"{cfg.get('run_tag', cfg.get('model_name', 'neo'))}_ablate_{suffix}"
tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  echo "== Training: configs/wt103/neo_20m.yaml | core=~10M | activation=${activation_name} | epochs=5 ==" >&2
  if [[ -n "$BACKEND" ]]; then
    python3 -u scripts/train.py --config "$tmp_cfg" --device "$DEVICE" --backend "$BACKEND"
  else
    python3 -u scripts/train.py --config "$tmp_cfg" --device "$DEVICE"
  fi
  rm -f "$tmp_cfg"
}

cd "$ROOT_DIR"

if [[ ! -f "$BASE_CFG" ]]; then
  echo "Config not found: $BASE_CFG" >&2
  exit 1
fi

for activation_name in "${ACTIVATIONS[@]}"; do
  activation_name="$(echo "$activation_name" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [[ -z "$activation_name" ]]; then
    continue
  fi
  case "$activation_name" in
    id3|id4|id5|tanh|gelu|none|identity)
      run_with_activation "$activation_name"
      ;;
    *)
      echo "Skipping unknown activation '$activation_name' (allowed: id3,id4,id5,tanh,gelu,none)" >&2
      ;;
  esac
done
