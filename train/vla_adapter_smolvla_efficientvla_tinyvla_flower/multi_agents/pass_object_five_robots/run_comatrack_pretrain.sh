#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}" \
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}" \
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-6}" \
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-1e-6}" \
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-5}" \
CLIP_EPS="${CLIP_EPS:-0.1}" \
TARGET_KL="${TARGET_KL:-0.01}" \
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}" \
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-1}" \
MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-0.1}" \
ALGO=comatrack "${SCRIPT_DIR}/run_baseline_pretrain.sh"
