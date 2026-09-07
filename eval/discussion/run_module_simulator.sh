#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

TASK="${TASK:-all}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/tmp/discussion_module_simulator.json}"
PLOT_OUTPUT="${PLOT_OUTPUT:-${REPO_ROOT}/tmp/discussion_module_simulator.png}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_module_simulator.py" \
  --task "${TASK}" \
  --output-json "${OUTPUT_JSON}" \
  "$@"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_module_simulator_results.py" \
  --input-json "${OUTPUT_JSON}" \
  --output "${PLOT_OUTPUT}"
