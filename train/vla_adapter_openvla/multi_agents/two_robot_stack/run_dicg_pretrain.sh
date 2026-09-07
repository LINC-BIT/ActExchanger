#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/TwoRobotStackCubeUR10e-v1/sft/panda_ur10e_panda_gripper/vla_adapter_openvla_sft/20260726-180924/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_openvla}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.05}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_openvla/multi_agents/two_robot_stack/dicg_pretrain.py
  --task-name TwoRobotStackCubeUR10e-v1
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --critic-warmup-rollouts 0
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --full-kl-coef 0.0
)

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${RESUME_USE_BEST_AGENT}" == "1" ]]; then
  CMD+=(--resume-use-best-agent)
fi

"${CMD[@]}"
