#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BC_SFT_CKPT_PATH="${BC_SFT_CKPT_PATH:-ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_smolvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-3e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-6}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-5}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.02}"
USE_VISION_LORA="${USE_VISION_LORA:-0}"
TRAIN_VISION_BACKBONE="${TRAIN_VISION_BACKBONE:-0}"
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-auto}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"

TINY_HIDDEN_DIM="${TINY_HIDDEN_DIM:-640}"
TINY_VISION_LAYERS="${TINY_VISION_LAYERS:-7}"
TINY_DECODER_LAYERS="${TINY_DECODER_LAYERS:-8}"
TINY_ATTENTION_HEADS="${TINY_ATTENTION_HEADS:-10}"
TINY_PATCH_SIZE="${TINY_PATCH_SIZE:-14}"
TINY_FFN_MULT="${TINY_FFN_MULT:-4}"
TINY_NUM_ACTION_BINS="${TINY_NUM_ACTION_BINS:-256}"
TINY_PROMPT_LENGTH="${TINY_PROMPT_LENGTH:-24}"

if [[ -z "${BC_SFT_CKPT_PATH}" ]]; then
  echo "BC_SFT_CKPT_PATH is empty. Set it to a compatible bc_pretrain checkpoint such as best_agent.pt or latest_agent.pt." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/comatrack_pretrain.py
  --task-name TwoRobotPickCube-v2
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --init-agent-path "${BC_SFT_CKPT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --critic-warmup-rollouts "${CRITIC_WARMUP_ROLLOUTS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --policy-mode native
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --full-kl-coef 0.0
  --attn-implementation sdpa
  --tiny-hidden-dim "${TINY_HIDDEN_DIM}"
  --tiny-vision-layers "${TINY_VISION_LAYERS}"
  --tiny-decoder-layers "${TINY_DECODER_LAYERS}"
  --tiny-attention-heads "${TINY_ATTENTION_HEADS}"
  --tiny-patch-size "${TINY_PATCH_SIZE}"
  --tiny-ffn-mult "${TINY_FFN_MULT}"
  --tiny-num-action-bins "${TINY_NUM_ACTION_BINS}"
  --tiny-prompt-length "${TINY_PROMPT_LENGTH}"
)

if [[ "${FREEZE_VLA_BACKBONE}" == "1" ]]; then
  CMD+=(--freeze-vla-backbone)
elif [[ "${FREEZE_VLA_BACKBONE}" == "auto" ]]; then
  if [[ "${MODEL_BACKBONE}" == "openvla" ]]; then
    CMD+=(--freeze-vla-backbone)
  fi
elif [[ "${FREEZE_VLA_BACKBONE}" != "0" ]]; then
  echo "FREEZE_VLA_BACKBONE must be one of: auto, 0, 1" >&2
  exit 1
fi

if [[ "${MODEL_BACKBONE}" == "openvla" ]]; then
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "MODEL_DIR is required when MODEL_BACKBONE=openvla." >&2
    exit 1
  fi
  CMD+=(--model-dir "${MODEL_DIR}")
  CMD+=(--use-vla-lora --lora-r 16 --lora-alpha 16 --lora-dropout 0.0)
fi

if [[ "${USE_VISION_LORA}" == "1" ]]; then
  CMD+=(--use-vision-lora)
fi

if [[ "${TRAIN_VISION_BACKBONE}" == "1" ]]; then
  CMD+=(--train-vision-backbone)
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${RESUME_USE_BEST_AGENT}" == "1" ]]; then
  CMD+=(--resume-use-best-agent)
fi

"${CMD[@]}"
