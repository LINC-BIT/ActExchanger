#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

XLSX_PATH="${XLSX_PATH:-${SCRIPT_DIR}/data/531 ablation.xlsx}"
OUT_PNG="${OUT_PNG:-${REPO_ROOT}/tmp/ablation.png}"
OUT_PDF="${OUT_PDF:-${REPO_ROOT}/tmp/ablation.pdf}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_ablation.py" \
  --xlsx "${XLSX_PATH}" \
  --out-png "${OUT_PNG}" \
  --out-pdf "${OUT_PDF}" \
  "$@"
