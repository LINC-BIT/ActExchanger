#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGO="${ALGO:-mat}"
case "${ALGO}" in
  mappo|ppo)
    ENTRY="${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber/mappo_online_rl.py"
    ;;
  mat|dicg|tgcnet|comatrack|magrpo|maporl|mpdf|maple)
    ENTRY="${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber/${ALGO}_online_rl.py"
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use one of: mappo mat dicg tgcnet comatrack magrpo maporl mpdf maple" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber"
if [[ "${ALGO}" == "mappo" || "${ALGO}" == "ppo" ]]; then
  exec "${SCRIPT_DIR}/run_mappo_online_rl.sh" "$@"
fi

INIT_AGENT_PATH="${INIT_AGENT_PATH:-}"
if [[ -z "${INIT_AGENT_PATH}" ]]; then
  PRETRAIN_ROOT="${PRETRAIN_ROOT:-}"
  if [[ -z "${PRETRAIN_ROOT}" ]]; then
    echo "Missing PRETRAIN_ROOT for ALGO=${ALGO}. Set it in run_${ALGO}_online_baseline.sh or pass INIT_AGENT_PATH explicitly." >&2
    exit 1
  fi
  if [[ -f "${PRETRAIN_ROOT}" ]]; then
    INIT_AGENT_PATH="${PRETRAIN_ROOT}"
  elif [[ -f "${PRETRAIN_ROOT}/best_agent.pt" ]]; then
    INIT_AGENT_PATH="${PRETRAIN_ROOT}/best_agent.pt"
  else
    shopt -s nullglob
    for candidate in "${PRETRAIN_ROOT}"/*/best_agent.pt; do
      INIT_AGENT_PATH="${candidate}"
    done
    shopt -u nullglob
  fi
  if [[ -z "${INIT_AGENT_PATH}" ]]; then
    echo "Missing pretrain checkpoint for ALGO=${ALGO}: ${PRETRAIN_ROOT}" >&2
    echo "Expected a checkpoint file, a run directory with best_agent.pt, or a root containing */best_agent.pt." >&2
    echo "Run run_${ALGO}_pretrain.sh first, or set INIT_AGENT_PATH explicitly." >&2
    exit 1
  fi
fi
MODEL_DIR="${MODEL_DIR:-}"
LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TOTAL_STEPS="${TOTAL_STEPS:-50000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
SAVE_INTERVAL_PER_ROLLOUT="${SAVE_INTERVAL_PER_ROLLOUT:-2}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-6}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
FUTURE_HEAD_LEARNING_RATE="${FUTURE_HEAD_LEARNING_RATE:-3e-6}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.05}"
ENT_COEF="${ENT_COEF:-0.0}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
MAX_TIME="${MAX_TIME:-}"
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-0}"
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-0}"
MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-}"

CMD=(
  "${PYTHON_BIN}" "${ENTRY}"
  --task-name PlaceCucumber-v1
  --robot-name panda_widowx_widowx
  --model-backbone mixed_tiny_vla_smolvla_efficientvla
  --image-size 112
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --max-episode-steps 200
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --total-steps "${TOTAL_STEPS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --save-interval-per-rollout "${SAVE_INTERVAL_PER_ROLLOUT}"
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
  --ent-coef "${ENT_COEF}"
  --full-kl-coef 0.0
  --attn-implementation sdpa
  --env-change-time-points "${ENV_CHANGE_TIME_POINTS}"
  --tiny-prompt-length 24
)

if [[ "${ALGO}" == "maple" ]]; then
  MAPLE_FUTURE_STATE_COEF="${MAPLE_FUTURE_STATE_COEF:-0.1}"
  MAPLE_LATENT_LAYERS="${MAPLE_LATENT_LAYERS:-2}"
  MAPLE_FUTURE_HORIZON="${MAPLE_FUTURE_HORIZON:-1}"
  MAPLE_RL_BC_COEF="${MAPLE_RL_BC_COEF:-0.0}"
  CMD+=(
    --maple-future-state-coef "${MAPLE_FUTURE_STATE_COEF}"
    --maple-latent-layers "${MAPLE_LATENT_LAYERS}"
    --maple-future-horizon "${MAPLE_FUTURE_HORIZON}"
    --maple-rl-bc-coef "${MAPLE_RL_BC_COEF}"
  )
fi

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
fi
if [[ -n "${LOW_LEVEL_AGENT_PATH}" ]]; then
  CMD+=(--low-level-agent-path "${LOW_LEVEL_AGENT_PATH}")
fi
if [[ -n "${MODEL_DIR}" ]]; then
  CMD+=(--model-dir "${MODEL_DIR}")
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
if [[ "${FREEZE_VLA_BACKBONE}" == "1" ]]; then
  CMD+=(--freeze-vla-backbone)
fi
if [[ -n "${MAGRPO_KL_COEF}" ]]; then
  CMD+=(--magrpo-kl-coef "${MAGRPO_KL_COEF}")
fi

"${CMD[@]}"
