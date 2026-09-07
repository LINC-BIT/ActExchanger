#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=dicg
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/dicg/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_dicg/20260716-104306/best_agent.pt}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
