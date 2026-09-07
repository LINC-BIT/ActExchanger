#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.png}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.json}"
CSV_OUTPUT="${CSV_OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.csv}"
INPUT_JSON="${INPUT_JSON:-}"
cd "${REPO_ROOT}"

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/plot_model_size_limit.py" --output "${OUTPUT}" --summary-output "${SUMMARY_OUTPUT}" --csv-output "${CSV_OUTPUT}")
if [[ -n "${INPUT_JSON}" ]]; then
  CMD+=(--input-json "${INPUT_JSON}")
fi
exec "${CMD[@]}" "$@"
