#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_smolvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.02}"
MAPORL_KL_COEF="${MAPORL_KL_COEF:-0.0}"
MAPORL_TEAM_ADV_COEF="${MAPORL_TEAM_ADV_COEF:-0.0}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
PLANNER_LOW_LEVEL_DETERMINISTIC="${PLANNER_LOW_LEVEL_DETERMINISTIC:-1}"
PLANNER_TEXT_MODE="${PLANNER_TEXT_MODE:-tiny_text}"
PLANNER_HIDDEN_DIM="${PLANNER_HIDDEN_DIM:-128}"
PLANNER_LAYERS="${PLANNER_LAYERS:-1}"

TINY_HIDDEN_DIM="${TINY_HIDDEN_DIM:-640}"
TINY_VISION_LAYERS="${TINY_VISION_LAYERS:-7}"
TINY_DECODER_LAYERS="${TINY_DECODER_LAYERS:-8}"
TINY_ATTENTION_HEADS="${TINY_ATTENTION_HEADS:-10}"
TINY_PATCH_SIZE="${TINY_PATCH_SIZE:-14}"
TINY_FFN_MULT="${TINY_FFN_MULT:-4}"
TINY_NUM_ACTION_BINS="${TINY_NUM_ACTION_BINS:-256}"
TINY_PROMPT_LENGTH="${TINY_PROMPT_LENGTH:-24}"

if [[ -z "${LOW_LEVEL_AGENT_PATH}" ]]; then
  echo "LOW_LEVEL_AGENT_PATH is required. Set it to a trained low-level VLA checkpoint such as best_agent.pt or latest_agent.pt." >&2
  exit 1
fi

if [[ "${MODEL_BACKBONE}" != "smolvla" && "${MODEL_BACKBONE}" != "mixed_tiny_vla_smolvla" ]]; then
  echo "MODEL_BACKBONE must be smolvla or mixed_tiny_vla_smolvla for planner pretraining." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/maporl_pretrain.py
  --task-name TwoRobotPickCube-v2
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --init-agent-path "${LOW_LEVEL_AGENT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
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
  --maporl-kl-coef "${MAPORL_KL_COEF}"
  --maporl-team-adv-coef "${MAPORL_TEAM_ADV_COEF}"
  --planner-text-mode "${PLANNER_TEXT_MODE}"
  --planner-hidden-dim "${PLANNER_HIDDEN_DIM}"
  --planner-layers "${PLANNER_LAYERS}"
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
