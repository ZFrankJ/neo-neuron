#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[Deprecated] scripts/ablation.sh has been renamed to scripts/ablation_norm.sh" >&2
exec "${ROOT_DIR}/scripts/ablation_norm.sh" "$@"
