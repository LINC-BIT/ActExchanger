#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=maporl
export SEED="${SEED:-2}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/maporl/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_maporl_planner/20260717-073213}"
export LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/toy_cnn_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"
export MAPORL_TEAM_ADV_COEF="${MAPORL_TEAM_ADV_COEF:-0.5}"
export MAPORL_KL_COEF="${MAPORL_KL_COEF:-0.05}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
