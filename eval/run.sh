#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MWE="${MWE:-0}"

run_step() {
  local name="$1"
  shift
  echo "[eval/run.sh] starting: ${name}"
  env MWE="${MWE}" "$@"
  echo "[eval/run.sh] finished: ${name}"
}

run_step "Experiment 1: accuracy comparison" \
  bash "${SCRIPT_DIR}/acc_comparison/run_acc_comparison.sh"

run_step "Experiment 2: overhead comparison" \
  env RUN_ACC_COMPARISON=0 bash "${SCRIPT_DIR}/overhead/run_overhead.sh"

run_step "Experiment 3: ablation" \
  bash "${SCRIPT_DIR}/ablation/run_ablation.sh"

run_step "Experiment 4: ICL ability" \
  bash "${SCRIPT_DIR}/discussion/run_discussion.sh" icl

run_step "Experiment 5: VLA versus CNN" \
  bash "${SCRIPT_DIR}/discussion/run_discussion.sh" vla-vs-cnn

run_step "Experiment 6: maximum supported model size" \
  bash "${SCRIPT_DIR}/discussion/run_discussion.sh" model-size

echo "[eval/run.sh] all Section 2 experiments finished"
