#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/PlaceCucumber-v1/sft/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_sft/20260709-054512/best_agent.pt}" \
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}" \
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-6}" \
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-1e-6}" \
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-5}" \
CLIP_EPS="${CLIP_EPS:-0.1}" \
TARGET_KL="${TARGET_KL:-0.01}" \
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-100}" \
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-1}" \
MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-0.1}" \
ALGO=comatrack "${SCRIPT_DIR}/run_baseline_pretrain.sh"
