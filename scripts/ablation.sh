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

L_VALUES=(0 1 2 3 5)

cd "$ROOT_DIR"

for L in "${L_VALUES[@]}"; do
  TMP_CFG="$(mktemp -t neo_ablation.XXXXXX.yaml)"
  cleanup() {
    rm -f "$TMP_CFG"
  }
  trap cleanup EXIT

  python3 - <<'PY' "$CFG_PATH" "$TMP_CFG" "$L"
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required for ablation runs. Install with: pip install pyyaml") from exc

src = Path(sys.argv[1])
out = Path(sys.argv[2])
L = float(sys.argv[3])

with src.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

cfg["g_clamp_L"] = L
tag_suffix = "noclamp" if L <= 0 else f"gclamp_{int(L) if L.is_integer() else L}"
cfg["run_tag"] = f"{cfg.get('run_tag', 'wt103_neo_20m')}_{tag_suffix}"
cfg["resume_path"] = ""
cfg["restart_done"] = False

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

  python3 scripts/train.py --config "$TMP_CFG" --device "$DEVICE"

  rm -f "$TMP_CFG"
done
