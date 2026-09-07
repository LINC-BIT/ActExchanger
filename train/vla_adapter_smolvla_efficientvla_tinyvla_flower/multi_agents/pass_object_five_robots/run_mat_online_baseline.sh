#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=mat
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/mat/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mat}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
