#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/TwoRobotPickCube-v2/comatrack/pandas_pandas/vla_adapter_smolvla_mappo/20260701-073031/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_smolvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
SAVE_INTERVAL_PER_ROLLOUT="${SAVE_INTERVAL_PER_ROLLOUT:-2}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-0}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-6}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.02}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"
MAX_TIME="${MAX_TIME:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
POLICY_MODE="${POLICY_MODE:-native}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"

if [[ -z "${INIT_AGENT_PATH}" ]]; then
  echo "INIT_AGENT_PATH is empty. Set it to a mixed PPO checkpoint such as latest_agent.pt." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/comatrack_online_rl.py
  --task-name TwoRobotPickCube-v2
  --baseline
  --model-backbone "${MODEL_BACKBONE}"
  --model-dir "${MODEL_DIR}"
  --image-size 112
  --init-agent-path "${INIT_AGENT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --save-interval-per-rollout "${SAVE_INTERVAL_PER_ROLLOUT}"
  --critic-warmup-rollouts "${CRITIC_WARMUP_ROLLOUTS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --policy-mode "${POLICY_MODE}"
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --full-kl-coef 0.0
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --env-change-time-points "${ENV_CHANGE_TIME_POINTS}"
)

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
