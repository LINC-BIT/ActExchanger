#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=tgcnet
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/tgcnet/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_tgcnet/20260716-104305/best_agent.pt}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
