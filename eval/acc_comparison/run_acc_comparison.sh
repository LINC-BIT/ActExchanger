#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS=(object_picking_placing object_stacking cucumber_placing cylinder_transfer)
if [[ -n "${WORKLOAD:-}" ]]; then
  workload="${WORKLOAD}"
else
  workload="${1:-}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
fi

if [[ -z "${workload}" || "${workload}" == "all" ]]; then
  for workload in "${WORKLOADS[@]}"; do
    echo "[run_acc_comparison] running ${workload}"
    bash "${SCRIPT_DIR}/${workload}/run_acc_comparison.sh" "$@"
  done
  exit 0
fi

case "${workload}" in
  object_picking_placing|object_stacking|cucumber_placing|cylinder_transfer)
    exec "${SCRIPT_DIR}/${workload}/run_acc_comparison.sh" "$@"
    ;;
  *)
    echo "Unknown workload: ${workload}" >&2
    echo "Supported workloads: ${WORKLOADS[*]}" >&2
    exit 2
    ;;
esac
