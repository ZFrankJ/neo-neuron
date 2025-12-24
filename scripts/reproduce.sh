#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TOKENIZERS_PARALLELISM=false

run_train() {
  local cfg_path="$1"
  local log_path
  log_path="$(mktemp -t neo_repro_log.XXXXXX.txt)"
  python3 scripts/train.py --config "$cfg_path" | tee "$log_path"
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
  save_dir="$(cfg_value "$cfg_path" save_dir)"
  run_tag="$(cfg_value "$cfg_path" run_tag)"

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --checkpoint "$save_dir/epoch_01_${run_tag}.pt" \
    --out-dir "$run_dir/probe/epoch01" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers 2 \
    --neurons-per-layer 256

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --checkpoint "$save_dir/best_${run_tag}.pt" \
    --out-dir "$run_dir/probe/best" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers 2 \
    --neurons-per-layer 256

  python3 scripts/probe.py \
    --config "$cfg_path" \
    --checkpoint "$save_dir/last_${run_tag}.pt" \
    --out-dir "$run_dir/probe/last" \
    --seq-len 1024 \
    --batch-size 1 \
    --start 0 \
    --layers 2 \
    --neurons-per-layer 256
}

cd "$ROOT_DIR"

# WT-103
run_dir_wt103_neo="$(run_train configs/wt103/neo_30m.yaml)"
eval_best configs/wt103/neo_30m.yaml
probe_wt103 configs/wt103/neo_30m.yaml "$run_dir_wt103_neo"

run_dir_wt103_lstm="$(run_train configs/wt103/lstm_30m.yaml)"
eval_best configs/wt103/lstm_30m.yaml
probe_wt103 configs/wt103/lstm_30m.yaml "$run_dir_wt103_lstm"

run_dir_wt103_tfm="$(run_train configs/wt103/transformer_30m.yaml)"
eval_best configs/wt103/transformer_30m.yaml

# WT-2
run_dir_wt2_neo="$(run_train configs/wt2/neo_6m.yaml)"
eval_best configs/wt2/neo_6m.yaml

run_dir_wt2_lstm_6m="$(run_train configs/wt2/lstm_6m.yaml)"
eval_best configs/wt2/lstm_6m.yaml

run_dir_wt2_lstm_25m="$(run_train configs/wt2/lstm_25m.yaml)"
eval_best configs/wt2/lstm_25m.yaml

python3 scripts/summarize_runs.py --root runs --out experiments/overall_summary
