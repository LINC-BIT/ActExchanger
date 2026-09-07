#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash eval/overhead/experiment.sh same-acc [same_acc.py options]
  bash eval/overhead/experiment.sh breakdown [benchmark_modules.py options]
  bash eval/overhead/experiment.sh plot [plot_module_time.py options]

same-acc with no options reads RUN_DIR, BASELINE_RUN_DIR, WORKLOAD,
and optional DICG_RUN_DIR, MAT_RUN_DIR, TGCNET_RUN_DIR, MAPLE_RUN_DIR,
COMATRACK_RUN_DIR, MAPORL_RUN_DIR, MPDF_RUN_DIR, and MAGRPO_RUN_DIR.
Only the four multi-agent workloads are accepted by same_acc.py.
EOF
}

command_name="${1:-help}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$command_name" in
  same-acc)
    if [[ "$#" -gt 0 ]]; then
      exec "$PYTHON_BIN" "$SCRIPT_DIR/same_acc.py" "$@"
    fi
    : "${WORKLOAD:?Set WORKLOAD or pass --workload}"
    : "${RUN_DIR:?Set RUN_DIR or pass --run-dir}"
    : "${BASELINE_RUN_DIR:?Set BASELINE_RUN_DIR or pass --baseline-run-dir}"
    args=(
      --workload "$WORKLOAD"
      --run-dir "$RUN_DIR"
      --baseline-run-dir "$BASELINE_RUN_DIR"
    )
    for method in DICG MAT TGCNET MAPLE COMATRACK MAPORL MPDF MAGRPO; do
      variable="${method}_RUN_DIR"
      if [[ -n "${!variable:-}" ]]; then
        flag="$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')"
        args+=("--${flag}-run-dir" "${!variable}")
      fi
    done
    exec "$PYTHON_BIN" "$SCRIPT_DIR/same_acc.py" "${args[@]}"
    ;;
  breakdown)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark_modules.py" "$@"
    ;;
  plot)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/plot_module_time.py" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    usage >&2
    exit 2
    ;;
esac
