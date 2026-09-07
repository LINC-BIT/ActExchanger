#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

WORKLOADS=(object_picking_placing object_stacking cucumber_placing cylinder_transfer)
METHODS=(DICG MAT TGCNET MAPLE COMATRACK MAPORL MPDF MAGRPO)

run_same_accuracy() {
  local workload="$1"
  local prefix="${workload^^}"
  local run_var="${prefix}_RUN_DIR"
  local baseline_var="${prefix}_BASELINE_RUN_DIR"
  local run_dir="${!run_var:-${RUN_DIR:-}}"
  local baseline_run_dir="${!baseline_var:-${BASELINE_RUN_DIR:-}}"
  local csv_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.csv"
  local json_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.json"
  local plot_output="${REPO_ROOT}/tmp/overhead_same_acc_${workload}.png"
  local args=(
    --workload "${workload}"
    --run-dir "${run_dir}"
    --baseline-run-dir "${baseline_run_dir}"
    --output-csv "${csv_output}"
    --output-json "${json_output}"
  )

  if [[ -z "${run_dir}" || -z "${baseline_run_dir}" ]]; then
    echo "Missing ${prefix}_RUN_DIR or ${prefix}_BASELINE_RUN_DIR" >&2
    exit 2
  fi

  for method in "${METHODS[@]}"; do
    local variable="${prefix}_${method}_RUN_DIR"
    local common_variable="${method}_RUN_DIR"
    local value="${!variable:-${!common_variable:-}}"
    if [[ -n "${value}" ]]; then
      args+=("--$(printf '%s' "${method}" | tr '[:upper:]' '[:lower:]')-run-dir" "${value}")
    fi
  done

  echo "[run_overhead] same-accuracy: ${workload}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/same_acc.py" "${args[@]}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_overhead_same_acc.py" \
    --input-csv "${csv_output}" \
    --output "${plot_output}"
}

for workload in "${WORKLOADS[@]}"; do
  run_same_accuracy "${workload}"
done

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
