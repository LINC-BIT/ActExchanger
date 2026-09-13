#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS=(object_picking_placing object_stacking cucumber_placing cylinder_transfer)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${WORKLOAD:-}" ]]; then
  workload="${WORKLOAD}"
else
  workload="${1:-}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
fi

has_metrics() {
  local run_dir="$1"
  if [[ "${run_dir}" != /* ]]; then run_dir="${REPO_ROOT}/${run_dir}"; fi
  [[ -f "${run_dir}/metrics.json" ]] || compgen -G "${run_dir}/tb/events.out.tfevents.*" > /dev/null
}

workload_has_metrics() {
  local workload="$1"
  local wrapper="${SCRIPT_DIR}/${workload}/run_acc_comparison.sh"
  local run_dir
  local baseline_run_dir
  run_dir="$(grep -m1 "^RUN_DIR=" "${wrapper}" | sed -E 's/^[^=]+="\$\{[^:}-]+:?-([^}]*)\}"/\1/')"
  baseline_run_dir="$(grep -m1 "^BASELINE_RUN_DIR=" "${wrapper}" | sed -E 's/^[^=]+="\$\{[^:}-]+:?-([^}]*)\}"/\1/')"
  has_metrics "${run_dir}" && has_metrics "${baseline_run_dir}"
}

if [[ -z "${workload}" || "${workload}" == "all" ]]; then
  for workload in "${WORKLOADS[@]}"; do
    echo "[run_acc_comparison] running ${workload}"
    if [[ "${MWE:-0}" == "1" ]] && ! workload_has_metrics "${workload}"; then
      echo "[run_acc_comparison] MWE: skipping ${workload}; checkpoint metrics are unavailable" >&2
      continue
    fi
    bash "${SCRIPT_DIR}/${workload}/run_acc_comparison.sh" "$@"
  done
  exit 0
fi

case "${workload}" in
  object_picking_placing|object_stacking|cucumber_placing|cylinder_transfer)
    if [[ "${MWE:-0}" == "1" ]] && ! workload_has_metrics "${workload}"; then
      echo "[run_acc_comparison] MWE: skipping ${workload}; checkpoint metrics are unavailable" >&2
      exit 0
    fi
    exec "${SCRIPT_DIR}/${workload}/run_acc_comparison.sh" "$@"
    ;;
  *)
    echo "Unknown workload: ${workload}" >&2
    echo "Supported workloads: ${WORKLOADS[*]}" >&2
    exit 2
    ;;
esac
