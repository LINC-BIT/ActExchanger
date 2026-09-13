#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.png}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.json}"
CSV_OUTPUT="${CSV_OUTPUT:-${REPO_ROOT}/tmp/discussion_model_size_limit.csv}"
INPUT_JSON="${INPUT_JSON:-}"
MODEL_SIZE_RUN_DIRS="${MODEL_SIZE_RUN_DIRS:-}"
MODEL_SIZE_DEVICE="${MODEL_SIZE_DEVICE:-cuda}"
MODEL_SIZE_BUDGET_GB="${MODEL_SIZE_BUDGET_GB:-32}"
cd "${REPO_ROOT}"

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/plot_model_size_limit.py" --output "${OUTPUT}" --summary-output "${SUMMARY_OUTPUT}" --csv-output "${CSV_OUTPUT}")
if [[ -n "${INPUT_JSON}" ]]; then
  CMD+=(--input-json "${INPUT_JSON}")
elif [[ "${MWE:-0}" != "1" && -n "${MODEL_SIZE_RUN_DIRS}" ]]; then
  measured_json="${REPO_ROOT}/tmp/discussion_model_size_measured.json"
  entries=()
  IFS=',' read -r -a run_dirs <<< "${MODEL_SIZE_RUN_DIRS}"
  for index in "${!run_dirs[@]}"; do
    entries+=(--entry "workload_${index}::${run_dirs[$index]}")
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/measure_model_size.py" "${entries[@]}" --output "${measured_json}" \
    --device "${MODEL_SIZE_DEVICE}" --budget-gb "${MODEL_SIZE_BUDGET_GB}"
  CMD+=(--input-json "${measured_json}")
elif [[ "${MWE:-0}" != "1" ]]; then
  echo "Full model-size run requires INPUT_JSON or MODEL_SIZE_RUN_DIRS." >&2
  exit 2
fi
exec "${CMD[@]}" "$@"
