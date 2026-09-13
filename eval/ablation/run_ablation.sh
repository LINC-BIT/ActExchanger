#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${REPO_ROOT}"

XLSX_PATH="${XLSX_PATH:-${SCRIPT_DIR}/data/531 ablation.xlsx}"
OUT_PNG="${OUT_PNG:-${REPO_ROOT}/tmp/ablation.png}"
OUT_PDF="${OUT_PDF:-${REPO_ROOT}/tmp/ablation.pdf}"
RESULTS_JSON="${RESULTS_JSON:-${REPO_ROOT}/tmp/ablation_results.json}"
WORKLOAD="${WORKLOAD:-object_picking_placing}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/tmp/ablation_runs/${WORKLOAD}}"
OURS_RUN_DIR="${OURS_RUN_DIR:-}"

mkdir -p "${RUN_ROOT}"

if [[ "${MWE:-0}" == "1" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_ablation.py" \
    --xlsx "${XLSX_PATH}" \
    --out-png "${OUT_PNG}" \
    --out-pdf "${OUT_PDF}" \
    "$@"
fi

if [[ -z "${OURS_RUN_DIR}" ]]; then
  wrapper="${REPO_ROOT}/eval/acc_comparison/${WORKLOAD}/run_acc_comparison.sh"
  if [[ ! -f "${wrapper}" ]]; then
    echo "Unknown workload: ${WORKLOAD}" >&2
    exit 2
  fi
  OURS_RUN_DIR="$(sed -nE 's/^RUN_DIR="\$\{RUN_DIR(:-|-)([^}]*)\}".*/\2/p' "${wrapper}" | head -n 1)"
  if [[ -z "${OURS_RUN_DIR}" ]]; then
    echo "Cannot determine ActExchanger reference run from ${wrapper}; set OURS_RUN_DIR." >&2
    exit 2
  fi
fi
if [[ "${OURS_RUN_DIR}" != /* ]]; then
  OURS_RUN_DIR="${REPO_ROOT}/${OURS_RUN_DIR}"
fi
if [[ ! -f "${OURS_RUN_DIR}/metrics.json" ]]; then
  echo "Missing ActExchanger reference metrics: ${OURS_RUN_DIR}/metrics.json" >&2
  exit 2
fi

run_variant() {
  local choice="$1"
  local output_dir="${RUN_ROOT}/$(printf '%s' "${choice}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd 'a-z0-9_')"
  if [[ ! -f "${output_dir}/metrics.json" ]]; then
    bash "${SCRIPT_DIR}/run_ablation_variant.sh" "${WORKLOAD}" "${choice}" "${output_dir}" > "${output_dir}.log"
  fi
  printf '%s' "${output_dir}"
}

original_length="$(run_variant "Original length")"
raw_features="$(run_variant "Raw features")"
semantic_features="$(run_variant "Semantic-only features")"
spatial_features="$(run_variant "Spatial-only features")"
multiple_vectors="$(run_variant "Multiple unfused features per action")"
no_aggregation="$(run_variant "No aggregation")"
current_action="$(run_variant "Current-forward aggregation")"
random_action="$(run_variant "Random aggregation")"

collect_args=(
  --run "Feature length::Original length::${original_length}"
  --run "Feature length::MDL::${OURS_RUN_DIR}"
  --run "Feature generation::Raw features::${raw_features}"
  --run "Feature generation::Semantic-only features::${semantic_features}"
  --run "Feature generation::Spatial-only features::${spatial_features}"
  --run "Feature generation::Semantic-spatial hybrid features::${OURS_RUN_DIR}"
  --run "Knowledge representation::Multiple unfused features per action::${multiple_vectors}"
  --run "Knowledge representation::Single fused feature per action::${OURS_RUN_DIR}"
  --run "Aggregation strategy::No aggregation::${no_aggregation}"
  --run "Aggregation strategy::Current-forward aggregation::${current_action}"
  --run "Aggregation strategy::Random aggregation::${random_action}"
  --run "Aggregation strategy::Same-action aggregation::${OURS_RUN_DIR}"
  --output-json "${RESULTS_JSON}"
)
"${PYTHON_BIN}" "${SCRIPT_DIR}/collect_ablation_results.py" "${collect_args[@]}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_ablation.py" \
  --results-json "${RESULTS_JSON}" \
  --out-png "${OUT_PNG}" \
  --out-pdf "${OUT_PDF}" \
  "$@"
