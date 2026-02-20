#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-auto}"
BACKEND="${2:-mlx}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

cd "$ROOT_DIR"

cfg_info() {
  local cfg_path="$1"
  python3 - <<'PY' "$cfg_path"
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
model = str(cfg.get("model_name", ""))
n_layers = int(cfg.get("n_layers", 1))
dataset = str(cfg.get("dataset_config", "dataset"))
save_dir = str(cfg.get("save_dir", "checkpoints"))
print(f"{model}|{n_layers}|{dataset}|{save_dir}")
PY
}

latest_run_dir() {
  local dataset="$1"
  local model="$2"
  local run_tag="$3"
  local base="${ROOT_DIR}/runs/${dataset}/${model}/${run_tag}"
  if [[ ! -d "$base" ]]; then
    return 1
  fi
  local latest
  latest="$(ls -td "$base"/* 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

probe_checkpoints_for_run() {
  local cfg_path="$1"
  local run_tag="$2"

  local info
  info="$(cfg_info "$cfg_path")"
  local model n_layers dataset save_dir
  IFS='|' read -r model n_layers dataset save_dir <<<"$info"

  local run_dir
  run_dir="$(latest_run_dir "$dataset" "$model" "$run_tag" || true)"
  if [[ -z "$run_dir" ]]; then
    echo "[Probe] Skip: run dir not found for ${model}/${run_tag}" >&2
    return 0
  fi

  local labels=("epoch_02" "epoch_04" "epoch_06" "best")
  for label in "${labels[@]}"; do
    local ckpt
    if [[ "$label" == "best" ]]; then
      ckpt="${save_dir}/best_${run_tag}.pt"
    else
      ckpt="${save_dir}/${label}_${run_tag}.pt"
    fi
    if [[ ! -f "$ckpt" ]]; then
      echo "[Probe] Skip: checkpoint not found: $ckpt" >&2
      continue
    fi

    local out_dir="${run_dir}/probe/${label}"
    echo "[Probe] ${model}/${run_tag} -> ${label}" >&2
    python3 scripts/probe.py \
      --config "$cfg_path" \
      --checkpoint "$ckpt" \
      --out-dir "$out_dir" \
      --seq-len 1024 \
      --batch-size 1 \
      --start 0 \
      --layers "$n_layers" \
      --neurons-per-layer 256 \
      --backend torch
  done
}

probe_scaling_runs() {
  local cfgs=(
    "configs/wt103/neo_20m.yaml"
    "configs/wt103/lstm_20m.yaml"
    "configs/wt103/neo_40m.yaml"
    "configs/wt103/lstm_40m.yaml"
    "configs/wt103/neo_80m.yaml"
    "configs/wt103/lstm_80m.yaml"
  )
  for cfg in "${cfgs[@]}"; do
    local stem
    stem="$(basename "$cfg" .yaml)"
    local run_tag="wt103_${stem}_scale_nonorm"
    probe_checkpoints_for_run "$cfg" "$run_tag"
  done
}

probe_ablation_runs() {
  local cfgs=(
    "configs/wt103/neo_20m.yaml"
    "configs/wt103/lstm_20m.yaml"
  )
  local suffixes=("ablate_nonorm" "ablate_rmsnorm_pre" "ablate_rmsnorm_all")
  for cfg in "${cfgs[@]}"; do
    local stem
    stem="$(basename "$cfg" .yaml)"
    for suffix in "${suffixes[@]}"; do
      local run_tag="wt103_${stem}_${suffix}"
      probe_checkpoints_for_run "$cfg" "$run_tag"
    done
  done
}

probe_variance_runs() {
  local cfgs=(
    "configs/wt103/neo_40m.yaml"
    "configs/wt103/lstm_40m.yaml"
  )
  local seeds=(42 1024 271828)
  for cfg in "${cfgs[@]}"; do
    local stem
    stem="$(basename "$cfg" .yaml)"
    for seed in "${seeds[@]}"; do
      local run_tag="wt103_${stem}_var_nonorm_seed${seed}"
      probe_checkpoints_for_run "$cfg" "$run_tag"
    done
  done
}

echo "== Stage 1: Scaling (Neo/LSTM 20/40/80M, no norm) ==" >&2
./scripts/scaling.sh "$DEVICE" "$BACKEND"
probe_scaling_runs

echo "== Stage 2: Ablation (Neo/LSTM 20M, none/rms-pre/rms-all) ==" >&2
./scripts/ablation.sh "$DEVICE" "$BACKEND" both "none,rms_pre,rms_all"
probe_ablation_runs

echo "== Stage 3: Variance (Neo/LSTM 40M, no norm, 3 seeds) ==" >&2
./scripts/variance.sh "$DEVICE" "$BACKEND"
probe_variance_runs

echo "== Stage 4: Transformer 40M training ==" >&2
if [[ -n "$BACKEND" ]]; then
  python3 -u scripts/train.py --config configs/wt103/transformer_40m.yaml --device "$DEVICE" --backend "$BACKEND"
else
  python3 -u scripts/train.py --config configs/wt103/transformer_40m.yaml --device "$DEVICE"
fi

python3 scripts/summarize_runs.py --root runs --out experiments/overall_summary
