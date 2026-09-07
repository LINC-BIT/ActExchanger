#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash eval/discussion/run_discussion.sh module-simulator [options]
  bash eval/discussion/run_discussion.sh vla-vs-cnn [options]
  bash eval/discussion/run_discussion.sh icl [options]
  bash eval/discussion/run_discussion.sh model-size [options]
  bash eval/discussion/run_discussion.sh all

module-simulator writes tmp/discussion_module_simulator.json by default.
vla-vs-cnn writes the 3h, 10h, and combined accuracy figures.
icl writes the four-workload ICL comparison figure and summary.
model-size writes the four-workload capacity and memory-overhead figures.
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "${command_name}" in
  module-simulator)
    exec "${SCRIPT_DIR}/run_module_simulator.sh" "$@"
    ;;
  vla-vs-cnn)
    exec "${SCRIPT_DIR}/run_vla_vs_cnn_accuracy.sh" "$@"
    ;;
  icl)
    exec "${SCRIPT_DIR}/run_icl.sh" "$@"
    ;;
  model-size)
    exec "${SCRIPT_DIR}/run_model_size_limit.sh" "$@"
    ;;
  all)
    "${SCRIPT_DIR}/run_module_simulator.sh"
    "${SCRIPT_DIR}/run_icl.sh"
    "${SCRIPT_DIR}/run_model_size_limit.sh"
    exec "${SCRIPT_DIR}/run_vla_vs_cnn_accuracy.sh"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown discussion experiment: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
