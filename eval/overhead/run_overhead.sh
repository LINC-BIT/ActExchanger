#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

WORKLOADS=(object_picking_placing object_stacking cucumber_placing cylinder_transfer)

RUN_ACC_COMPARISON="${RUN_ACC_COMPARISON:-1}"
SKIP_SAME_ACC="${SKIP_SAME_ACC:-${MWE:-0}}"

if [[ "${RUN_ACC_COMPARISON}" == "1" ]]; then
  echo "[run_overhead] running Experiment 1 accuracy comparison"
  bash "${REPO_ROOT}/eval/acc_comparison/run_acc_comparison.sh"
fi

run_same_accuracy() {
  local workload="$1"
  local csv_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.csv"
  local json_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.json"
  local plot_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.png"
  local args=(
    --workload "${workload}"
    --acc-comparison-wrapper "${REPO_ROOT}/eval/acc_comparison/${workload}/run_acc_comparison.sh"
    --output-csv "${csv_output}"
    --output-json "${json_output}"
  )

  echo "[run_overhead] same-accuracy: ${workload}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/same_acc.py" "${args[@]}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_overhead_same_acc.py" \
    --input-csv "${csv_output}" \
    --output "${plot_output}"
}

if [[ "${SKIP_SAME_ACC}" != "1" ]]; then
  for workload in "${WORKLOADS[@]}"; do
    run_same_accuracy "${workload}"
  done

fi
BREAKDOWN_JSON="${REPO_ROOT}/tmp/overhead_module_breakdown.json"
BREAKDOWN_PLOT="${REPO_ROOT}/tmp/overhead_module_breakdown.png"
BREAKDOWN_COLORED_PLOT="${REPO_ROOT}/tmp/overhead_module_breakdown_color.png"
BREAKDOWN_ARGS=(
  --workload all
  --output-json "${BREAKDOWN_JSON}"
)

if [[ "${MWE:-0}" == "1" ]]; then
  BREAKDOWN_ARGS+=(--warmup 2 --samples 5)
fi

echo "[run_overhead] module breakdown: all workloads"
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_modules.py" "${BREAKDOWN_ARGS[@]}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_module_time.py" \
  --input-json "${BREAKDOWN_JSON}" \
  --output "${BREAKDOWN_PLOT}" \
  --colored-output "${BREAKDOWN_COLORED_PLOT}"
