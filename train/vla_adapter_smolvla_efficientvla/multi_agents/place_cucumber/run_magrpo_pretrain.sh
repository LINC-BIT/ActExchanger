#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-${INIT_AGENT_PATH:-ckpt/PlaceCucumber-v1/sft/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_sft/20260709-054512/best_agent.pt}}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
TOTAL_STEPS="${TOTAL_STEPS:-50000000}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.05}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
PLANNER_LOW_LEVEL_DETERMINISTIC="${PLANNER_LOW_LEVEL_DETERMINISTIC:-1}"
PLANNER_TEXT_MODE="${PLANNER_TEXT_MODE:-fixed}"
PLANNER_HIDDEN_DIM="${PLANNER_HIDDEN_DIM:-256}"
PLANNER_LAYERS="${PLANNER_LAYERS:-2}"
MAGRPO_EPS="${MAGRPO_EPS:-1e-5}"
MAGRPO_GAMMA="${MAGRPO_GAMMA:-1.0}"
MAGRPO_GROUP_SIZE="${MAGRPO_GROUP_SIZE:-0}"
MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-0.05}"
MAGRPO_ENT_COEF="${MAGRPO_ENT_COEF:-0.0}"

TINY_HIDDEN_DIM="${TINY_HIDDEN_DIM:-384}"
TINY_VISION_LAYERS="${TINY_VISION_LAYERS:-4}"
TINY_DECODER_LAYERS="${TINY_DECODER_LAYERS:-4}"
TINY_ATTENTION_HEADS="${TINY_ATTENTION_HEADS:-6}"
TINY_PATCH_SIZE="${TINY_PATCH_SIZE:-14}"
TINY_FFN_MULT="${TINY_FFN_MULT:-4}"
TINY_NUM_ACTION_BINS="${TINY_NUM_ACTION_BINS:-256}"
TINY_PROMPT_LENGTH="${TINY_PROMPT_LENGTH:-24}"

SMOLVLA_HIDDEN_DIM="${SMOLVLA_HIDDEN_DIM:-384}"
SMOLVLA_VISION_LAYERS="${SMOLVLA_VISION_LAYERS:-6}"
SMOLVLA_ATTENTION_HEADS="${SMOLVLA_ATTENTION_HEADS:-6}"
SMOLVLA_PATCH_SIZE="${SMOLVLA_PATCH_SIZE:-14}"
SMOLVLA_FFN_MULT="${SMOLVLA_FFN_MULT:-4}"

EFFICIENTVLA_HIDDEN_DIM="${EFFICIENTVLA_HIDDEN_DIM:-256}"
EFFICIENTVLA_VISION_LAYERS="${EFFICIENTVLA_VISION_LAYERS:-4}"
EFFICIENTVLA_DECODER_LAYERS="${EFFICIENTVLA_DECODER_LAYERS:-1}"
EFFICIENTVLA_ATTENTION_HEADS="${EFFICIENTVLA_ATTENTION_HEADS:-4}"
EFFICIENTVLA_PATCH_SIZE="${EFFICIENTVLA_PATCH_SIZE:-16}"
EFFICIENTVLA_FFN_MULT="${EFFICIENTVLA_FFN_MULT:-3}"

if [[ -z "${LOW_LEVEL_AGENT_PATH}" ]]; then
  echo "LOW_LEVEL_AGENT_PATH is required for planner pretraining." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber/magrpo_pretrain.py
  --task-name PlaceCucumber-v1
  --robot-name panda_widowx_widowx
  --model-backbone mixed_tiny_vla_smolvla_efficientvla
  --image-size 112
  --low-level-agent-path "${LOW_LEVEL_AGENT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --max-episode-steps 200
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --total-steps "${TOTAL_STEPS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --policy-mode native
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --planner-text-mode "${PLANNER_TEXT_MODE}"
  --planner-hidden-dim "${PLANNER_HIDDEN_DIM}"
  --planner-layers "${PLANNER_LAYERS}"
  --magrpo-eps "${MAGRPO_EPS}"
  --magrpo-gamma "${MAGRPO_GAMMA}"
  --magrpo-group-size "${MAGRPO_GROUP_SIZE}"
  --magrpo-kl-coef "${MAGRPO_KL_COEF}"
  --magrpo-ent-coef "${MAGRPO_ENT_COEF}"
  --attn-implementation sdpa
  --tiny-hidden-dim "${TINY_HIDDEN_DIM}"
  --tiny-vision-layers "${TINY_VISION_LAYERS}"
  --tiny-decoder-layers "${TINY_DECODER_LAYERS}"
  --tiny-attention-heads "${TINY_ATTENTION_HEADS}"
  --tiny-patch-size "${TINY_PATCH_SIZE}"
  --tiny-ffn-mult "${TINY_FFN_MULT}"
  --tiny-num-action-bins "${TINY_NUM_ACTION_BINS}"
  --tiny-prompt-length "${TINY_PROMPT_LENGTH}"
  --smolvla-hidden-dim "${SMOLVLA_HIDDEN_DIM}"
  --smolvla-vision-layers "${SMOLVLA_VISION_LAYERS}"
  --smolvla-attention-heads "${SMOLVLA_ATTENTION_HEADS}"
  --smolvla-patch-size "${SMOLVLA_PATCH_SIZE}"
  --smolvla-ffn-mult "${SMOLVLA_FFN_MULT}"
  --efficientvla-hidden-dim "${EFFICIENTVLA_HIDDEN_DIM}"
  --efficientvla-vision-layers "${EFFICIENTVLA_VISION_LAYERS}"
  --efficientvla-decoder-layers "${EFFICIENTVLA_DECODER_LAYERS}"
  --efficientvla-attention-heads "${EFFICIENTVLA_ATTENTION_HEADS}"
  --efficientvla-patch-size "${EFFICIENTVLA_PATCH_SIZE}"
  --efficientvla-ffn-mult "${EFFICIENTVLA_FFN_MULT}"
)

if [[ -n "${MODEL_DIR}" ]]; then
  CMD+=(--model-dir "${MODEL_DIR}")
fi

if [[ "${PLANNER_LOW_LEVEL_DETERMINISTIC}" == "0" ]]; then
  CMD+=(--no-planner-low-level-deterministic)
else
  CMD+=(--planner-low-level-deterministic)
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${RESUME_USE_BEST_AGENT}" == "1" ]]; then
  CMD+=(--resume-use-best-agent)
fi

"${CMD[@]}"
