#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=mpdf
export SEED="${SEED:-3}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PassObjectFiveRobots-v1/mpdf/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mpdf_planner/20260717-073213}"
export LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/toy_cnn_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"
export MPDF_GROUP_SIZE="${MPDF_GROUP_SIZE:-8}"
export MPDF_RANK_TEMPERATURE="${MPDF_RANK_TEMPERATURE:-0.5}"
export MPDF_KL_COEF="${MPDF_KL_COEF:-0.05}"
export MPDF_ENT_COEF="${MPDF_ENT_COEF:-0.0}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
