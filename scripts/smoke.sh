#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${1:-cpu}"
export TOKENIZERS_PARALLELISM=false

CFG_LIST=(
  "${ROOT_DIR}/configs/wt2/neo_6m.yaml"
#  "${ROOT_DIR}/configs/wt2/lstm_6m.yaml"
#  "${ROOT_DIR}/configs/wt2/lstm_25m.yaml"
#  "${ROOT_DIR}/configs/wt103/lstm_30m.yaml"
#  "${ROOT_DIR}/configs/wt103/neo_30m.yaml"
#  "${ROOT_DIR}/configs/wt103/transformer_30m.yaml"
)

cd "$ROOT_DIR"

for CFG_PATH in "${CFG_LIST[@]}"; do
  if [[ ! -f "$CFG_PATH" ]]; then
    echo "Config not found: $CFG_PATH" >&2
    exit 1
  fi

  TMP_CFG="$(mktemp -t neo_smoke.XXXXXX.yaml)"
  LOG_PATH="$(mktemp -t neo_smoke_log.XXXXXX.txt)"
  cleanup() {
    rm -f "$TMP_CFG" "$LOG_PATH"
  }
  trap cleanup EXIT

  python3 - <<'PY' "$CFG_PATH" "$TMP_CFG"
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required for smoke runs. Install with: pip install pyyaml") from exc

src = Path(sys.argv[1])
out = Path(sys.argv[2])
with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

cfg["epochs"] = 1
cfg["run_tag"] = "smoke"
cfg["batch_size"] = min(int(cfg.get("batch_size", 8)), 4)
cfg["block_size"] = min(int(cfg.get("block_size", 64)), 64)
if "model_name" not in cfg:
    if "n_heads" in cfg and "ff_mult" in cfg:
        cfg["model_name"] = "transformer"
    elif "cell_type" in cfg:
        cfg["model_name"] = "neo"
    else:
        cfg["model_name"] = "lstm"

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

  python3 scripts/train.py --config "$TMP_CFG" --device "$DEVICE" | tee "$LOG_PATH"

  RUN_DIR="$(sed -n 's/^Run artifacts saved to //p' "$LOG_PATH" | tail -n1)"
  if [[ -z "$RUN_DIR" ]]; then
    echo "Failed to locate run directory in train output." >&2
    exit 1
  fi

  SAVE_DIR="$(python3 - <<'PY' "$TMP_CFG"
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(cfg.get("save_dir", "checkpoints"))
PY
)"

  python3 scripts/eval.py --config "$TMP_CFG" --checkpoint "$SAVE_DIR/best_smoke.pt" --split test --device "$DEVICE"

  python3 scripts/probe.py \
    --config "$TMP_CFG" \
    --checkpoint "$SAVE_DIR/best_smoke.pt" \
    --out-dir "$RUN_DIR/probe" \
    --seq-len 64 \
    --layers 1 \
    --neurons-per-layer 2 \
    --device "$DEVICE"

  rm -f "$TMP_CFG" "$LOG_PATH"
done
