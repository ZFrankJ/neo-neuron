#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

run_train() {
  local cfg_path="$1"
  local log_path
  log_path="$(mktemp -t neo_repro_log.XXXXXX.txt)"
  python3 -u scripts/train.py --config "$cfg_path" 2>&1 | tee "$log_path" >&2
  local run_dir
  run_dir="$(sed -n 's/^Run artifacts saved to //p' "$log_path" | tail -n1)"
  rm -f "$log_path"
  if [[ -z "$run_dir" ]]; then
    echo "Failed to locate run directory for $cfg_path" >&2
    exit 1
  fi
  echo "$run_dir"
}

cfg_value() {
  python3 - <<'PY' "$1" "$2"
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(cfg.get(sys.argv[2], ""))
PY
}

eval_best() {
  local cfg_path="$1"
  local save_dir
  local run_tag
  save_dir="$(cfg_value "$cfg_path" save_dir)"
  run_tag="$(cfg_value "$cfg_path" run_tag)"
  python3 scripts/eval.py --config "$cfg_path" --checkpoint "$save_dir/best_${run_tag}.pt" --split test
}

probe_wt103() {
  local cfg_path="$1"
  local run_dir="$2"
  local save_dir
  local run_tag
  local n_layers
  save_dir="$(cfg_value "$cfg_path" save_dir)"
  run_tag="$(cfg_value "$cfg_path" run_tag)"
  n_layers="$(cfg_value "$cfg_path" n_layers)"
  if [[ -z "$n_layers" ]]; then
    n_layers=1
  fi

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --backend torch \
    --checkpoint "$save_dir/epoch_01_${run_tag}.pt" \
    --out-dir "$run_dir/probe/epoch01" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers "$n_layers" \
    --neurons-per-layer 256

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --backend torch \
    --checkpoint "$save_dir/best_${run_tag}.pt" \
    --out-dir "$run_dir/probe/best" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers "$n_layers" \
    --neurons-per-layer 256

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --backend torch \
    --checkpoint "$save_dir/last_${run_tag}.pt" \
    --out-dir "$run_dir/probe/last" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers "$n_layers" \
    --neurons-per-layer 256
}

cd "$ROOT_DIR"

# WT-103 in requested order:
# 1) two 10M models
run_dir_wt103_neo_10m="$(run_train configs/wt103/neo_10m.yaml)"
eval_best configs/wt103/neo_10m.yaml
probe_wt103 configs/wt103/neo_10m.yaml "$run_dir_wt103_neo_10m"

run_dir_wt103_lstm_10m="$(run_train configs/wt103/lstm_10m.yaml)"
eval_best configs/wt103/lstm_10m.yaml
probe_wt103 configs/wt103/lstm_10m.yaml "$run_dir_wt103_lstm_10m"

# 2) three 20M models
run_dir_wt103_neo_20m="$(run_train configs/wt103/neo_20m.yaml)"
eval_best configs/wt103/neo_20m.yaml
probe_wt103 configs/wt103/neo_20m.yaml "$run_dir_wt103_neo_20m"

run_dir_wt103_lstm_20m="$(run_train configs/wt103/lstm_20m.yaml)"
eval_best configs/wt103/lstm_20m.yaml
probe_wt103 configs/wt103/lstm_20m.yaml "$run_dir_wt103_lstm_20m"

run_dir_wt103_tfm_20m="$(run_train configs/wt103/transformer_20m.yaml)"
eval_best configs/wt103/transformer_20m.yaml

# 3) two 40M models
run_dir_wt103_neo_40m="$(run_train configs/wt103/neo_40m.yaml)"
eval_best configs/wt103/neo_40m.yaml
probe_wt103 configs/wt103/neo_40m.yaml "$run_dir_wt103_neo_40m"

run_dir_wt103_lstm_40m="$(run_train configs/wt103/lstm_40m.yaml)"
eval_best configs/wt103/lstm_40m.yaml
probe_wt103 configs/wt103/lstm_40m.yaml "$run_dir_wt103_lstm_40m"

python3 scripts/summarize_runs.py --root runs --out experiments/overall_summary
