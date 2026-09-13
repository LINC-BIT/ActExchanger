#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLA_RUN_DIR="${VLA_RUN_DIR:-}"
CNN_RUN_DIR="${CNN_RUN_DIR:-}"
OUTPUT_3H="${OUTPUT_3H:-${REPO_ROOT}/tmp/vla_vs_cnn_accuracy_3h.png}"
OUTPUT_10H="${OUTPUT_10H:-${REPO_ROOT}/tmp/vla_vs_cnn_accuracy_10h.png}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${REPO_ROOT}/tmp/vla_vs_cnn_accuracy_summary.json}"
cd "${REPO_ROOT}"

if [[ "${MWE:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_vla_vs_cnn_accuracy.py" "$@"
fi
if [[ -z "${VLA_RUN_DIR}" || -z "${CNN_RUN_DIR}" ]]; then
  echo "Full VLA-vs-CNN run requires VLA_RUN_DIR and CNN_RUN_DIR." >&2
  exit 2
fi
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_vla_vs_cnn_runs.py" \
  --vla-run-dir "${VLA_RUN_DIR}" --cnn-run-dir "${CNN_RUN_DIR}" \
  --output-3h "${OUTPUT_3H}" --output-10h "${OUTPUT_10H}" \
  --summary-output "${SUMMARY_OUTPUT}" "$@"
