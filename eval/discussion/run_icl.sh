#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/discussion_icl_multi_agent.png}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${REPO_ROOT}/tmp/discussion_icl_multi_agent.json}"
INPUT_JSON="${INPUT_JSON:-}"
POINTS="${POINTS:-30}"
cd "${REPO_ROOT}"

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/plot_icl_multi_agent.py" --output "${OUTPUT}" --summary-output "${SUMMARY_OUTPUT}" --points "${POINTS}")
if [[ -n "${INPUT_JSON}" ]]; then
  CMD+=(--input-json "${INPUT_JSON}")
fi
exec "${CMD[@]}" "$@"
