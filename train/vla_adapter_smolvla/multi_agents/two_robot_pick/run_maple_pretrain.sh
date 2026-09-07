#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/TwoRobotPickCube-v2/sft/pandas_pandas/vla_adapter_smolvla_sft/20260628-151306/best_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_smolvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-3e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-6}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
FUTURE_HEAD_LEARNING_RATE="${FUTURE_HEAD_LEARNING_RATE:-3e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
CLIP_EPS="${CLIP_EPS:-0.2}"
TARGET_KL="${TARGET_KL:-0.2}"
MAPLE_FUTURE_STATE_COEF="${MAPLE_FUTURE_STATE_COEF:-0.1}"
MAPLE_LATENT_LAYERS="${MAPLE_LATENT_LAYERS:-2}"
MAPLE_FUTURE_HORIZON="${MAPLE_FUTURE_HORIZON:-1}"
MAPLE_RL_BC_COEF="${MAPLE_RL_BC_COEF:-0.0}"
RESUME_DIR="${RESUME_DIR:-}"

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/maple_pretrain.py
  --task-name TwoRobotPickCube-v2
  --model-backbone "${MODEL_BACKBONE}"
  --model-dir "${MODEL_DIR}"
  --image-size 112
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
  --use-vla-lora
  --lora-r 16
  --lora-alpha 16
  --lora-dropout 0.05
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --future-head-learning-rate "${FUTURE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --maple-future-state-coef "${MAPLE_FUTURE_STATE_COEF}"
  --maple-latent-layers "${MAPLE_LATENT_LAYERS}"
  --maple-future-horizon "${MAPLE_FUTURE_HORIZON}"
  --maple-rl-bc-coef "${MAPLE_RL_BC_COEF}"
  --attn-implementation sdpa
)

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

"${CMD[@]}"
