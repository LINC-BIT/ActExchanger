#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/TwoRobotStackCubeUR10e-v1/sft/panda_ur10e_panda_gripper/vla_adapter_openvla_sft/20260726-180924/latest_agent.pt}"
INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/TwoRobotStackCubeUR10e-v1/maporl/panda_ur10e_panda_gripper/vla_adapter_openvla_maporl_planner/20260728-111514/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_openvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
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
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"
MAX_TIME="${MAX_TIME:-}"
PLANNER_LOW_LEVEL_DETERMINISTIC="${PLANNER_LOW_LEVEL_DETERMINISTIC:-1}"
PLANNER_TEXT_MODE="${PLANNER_TEXT_MODE:-tiny_text}"
PLANNER_HIDDEN_DIM="${PLANNER_HIDDEN_DIM:-128}"
PLANNER_LAYERS="${PLANNER_LAYERS:-1}"

if [[ -z "${LOW_LEVEL_AGENT_PATH}" ]]; then
  echo "LOW_LEVEL_AGENT_PATH is required." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_openvla/multi_agents/two_robot_stack/maporl_online_rl.py
  --task-name TwoRobotStackCubeUR10e-v1
  --model-backbone "${MODEL_BACKBONE}"
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
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --maporl-kl-coef "${MAPORL_KL_COEF}"
  --maporl-team-adv-coef "${MAPORL_TEAM_ADV_COEF}"
  --low-level-agent-path "${LOW_LEVEL_AGENT_PATH}"
  --planner-text-mode "${PLANNER_TEXT_MODE}"
  --planner-hidden-dim "${PLANNER_HIDDEN_DIM}"
  --planner-layers "${PLANNER_LAYERS}"
  --env-change-time-points "${ENV_CHANGE_TIME_POINTS}"
)

if [[ -n "${MODEL_DIR}" ]]; then
  CMD+=(--model-dir "${MODEL_DIR}")
fi

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
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

if [[ -n "${MAX_TIME}" ]]; then
  CMD+=(--max-time "${MAX_TIME}")
fi

"${CMD[@]}"
