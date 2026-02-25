#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-mlx}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SEEDS=(42 1024 271828)
BASE_CFGS=(
  "${ROOT_DIR}/configs/wt103/neo_40m.yaml"
  "${ROOT_DIR}/configs/wt103/lstm_40m.yaml"
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
import math
import re

def _target_from_path(path: Path, run_tag: str) -> int | None:
    m = re.search(r"_(\d+)m$", path.stem)
    if m:
        return int(m.group(1)) * 1_000_000
    m = re.search(r"(\d+)m", run_tag or "")
    if m:
        return int(m.group(1)) * 1_000_000
    return None

def _norm_kind(norm: str) -> str:
    n = str(norm).strip().lower()
    if n in ("layernorm", "layer_norm", "ln"):
        return "layernorm"
    if n in ("rmsnorm", "rms_norm", "rms"):
        return "rmsnorm"
    return "none"

def _norm_place_count(place: str, n_layers: int) -> int:
    p = str(place).strip().lower()
    if p == "pre":
        return n_layers
    if p == "stack":
        return 1
    return n_layers + 1

def _retune_d_model(cfg: dict, target: int | None) -> int:
    model = str(cfg.get("model_name", "")).strip().lower()
    if target is None or model not in ("neo", "lstm"):
        return int(cfg.get("d_model", 1))

    L = int(cfg.get("n_layers", 1))
    E = int(cfg.get("d_embed", 0))
    V = int(cfg.get("vocab_size", 0))
    norm = _norm_kind(str(cfg.get("recurrent_norm", "none")))

    if model == "neo":
        a = 2 * L
        b = 2 * E + 2 * L + 1
    else:
        a = 8 * L
        b = 2 * E + 8 * L + 1

    norm_count = _norm_place_count(str(cfg.get("recurrent_norm_place", "all")), L)
    if norm == "layernorm":
        b += 2 * norm_count
    elif norm == "rmsnorm":
        b += norm_count

    c = V * E + V + E
    disc = b * b - 4 * a * (c - target)
    if disc <= 0:
        return int(cfg.get("d_model", 1))

    d0 = max(1, int((-b + math.sqrt(disc)) / (2 * a)))
    lo = max(1, d0 - 4096)
    hi = d0 + 4096

    best_d = d0
    best_diff = None
    for d in range(lo, hi + 1):
        p = a * d * d + b * d + c
        diff = abs(p - target)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_d = d
    return best_d

base_cfg = Path(sys.argv[1])
tmp_cfg = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = yaml.safe_load(base_cfg.read_text()) or {}
cfg["recurrent_norm"] = "none"
cfg["recurrent_norm_place"] = "all"
cfg["seed"] = seed
cfg["resume_path"] = ""
cfg["run_tag"] = f"{cfg.get('run_tag', cfg.get('model_name', 'model'))}_var_nonorm_seed{seed}"
target = _target_from_path(base_cfg, cfg.get("run_tag", ""))
cfg["d_model"] = _retune_d_model(cfg, target)
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
