#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="${ROOT_DIR}/notebooks"
MODE="${1:-copy}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/notebooks_local.sh copy
    Create missing .local.ipynb copies beside tracked notebooks.

  ./scripts/notebooks_local.sh refresh
    Overwrite all .local.ipynb copies from the tracked notebooks.

  ./scripts/notebooks_local.sh run-all
    Create missing .local.ipynb copies, then execute each local notebook in place.

Notes:
  - Tracked notebooks remain unchanged.
  - Local execution copies are ignored by git via notebooks/*.local.ipynb.
  - Requires jupyter to be installed for run-all.
EOF
}

copy_missing() {
  for src in "${NOTEBOOK_DIR}"/*.ipynb; do
    [[ "$src" == *.local.ipynb ]] && continue
    local_path="${src%.ipynb}.local.ipynb"
    if [[ ! -f "$local_path" ]]; then
      cp "$src" "$local_path"
      echo "Created ${local_path#${ROOT_DIR}/}"
    fi
  done
}

refresh_all() {
  for src in "${NOTEBOOK_DIR}"/*.ipynb; do
    [[ "$src" == *.local.ipynb ]] && continue
    local_path="${src%.ipynb}.local.ipynb"
    cp "$src" "$local_path"
    echo "Refreshed ${local_path#${ROOT_DIR}/}"
  done
}

run_all() {
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "jupyter is required for run-all" >&2
    exit 1
  fi
  copy_missing
  for nb in "${NOTEBOOK_DIR}"/*.local.ipynb; do
    echo "Running ${nb#${ROOT_DIR}/}"
    jupyter nbconvert \
      --to notebook \
      --execute \
      --inplace \
      "$nb"
  done
}

cd "$ROOT_DIR"

case "$MODE" in
  copy)
    copy_missing
    ;;
  refresh)
    refresh_all
    ;;
  run-all)
    run_all
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac
