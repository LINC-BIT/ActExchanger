#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/discussion_icl_multi_agent.png}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${REPO_ROOT}/tmp/discussion_icl_multi_agent.json}"
INPUT_JSON="${INPUT_JSON:-}"
POINTS="${POINTS:-30}"
WORKLOADS=(object_picking_placing object_stacking cucumber_placing cylinder_transfer)
cd "${REPO_ROOT}"

if [[ -z "${INPUT_JSON}" && "${MWE:-0}" != "1" ]]; then
  COLLECTED_JSON="${REPO_ROOT}/tmp/discussion_icl_histories.json"
  collect_args=()
  for workload in "${WORKLOADS[@]}"; do
    wrapper="${REPO_ROOT}/eval/acc_comparison/${workload}/run_acc_comparison.sh"
    ours="$(sed -nE 's/^RUN_DIR="\$\{RUN_DIR(:-|-)([^}]*)\}".*/\2/p' "${wrapper}" | head -n 1)"
    variable="RICL_RUN_DIR_$(printf '%s' "${workload}" | tr '[:lower:]' '[:upper:]')"
    ricl="${!variable:-}"
    if [[ -z "${ricl}" ]]; then
      echo "Full ICL run requires ${variable}." >&2
      exit 2
    fi
    [[ "${ours}" = /* ]] || ours="${REPO_ROOT}/${ours}"
    [[ "${ricl}" = /* ]] || ricl="${REPO_ROOT}/${ricl}"
    collect_args+=(--entry "${workload}::${ours}::${ricl}")
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_icl_histories.py" "${collect_args[@]}" --output "${COLLECTED_JSON}"
  INPUT_JSON="${COLLECTED_JSON}"
fi

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/plot_icl_multi_agent.py" --output "${OUTPUT}" --summary-output "${SUMMARY_OUTPUT}" --points "${POINTS}")
if [[ -n "${INPUT_JSON}" ]]; then
  CMD+=(--input-json "${INPUT_JSON}")
fi
exec "${CMD[@]}" "$@"
