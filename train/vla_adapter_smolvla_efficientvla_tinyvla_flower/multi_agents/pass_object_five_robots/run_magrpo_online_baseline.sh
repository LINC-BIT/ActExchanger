#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=magrpo
export SEED="${SEED:-1}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/magrpo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_magrpo_planner/20260717-073220}"
export LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/toy_cnn_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"
export MAGRPO_GROUP_SIZE="${MAGRPO_GROUP_SIZE:-8}"
export MAGRPO_GAMMA="${MAGRPO_GAMMA:-1.0}"
export MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-0.05}"
export MAGRPO_ENT_COEF="${MAGRPO_ENT_COEF:-0.0}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
