#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MAPPO_CKPT_PATH="${MAPPO_CKPT_PATH:-ckpt/TwoRobotPickCube-v2/ppo/pandas_pandas/vla_adapter_smolvla_mappo/20260628-184332/latest_agent.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-mixed_tiny_vla_smolvla}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-100}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-3e-6}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
ACTOR_HOOK_LEARNING_RATE="${ACTOR_HOOK_LEARNING_RATE:-3e-6}"
FEATURE_AGGREGATOR_LEARNING_RATE="${FEATURE_AGGREGATOR_LEARNING_RATE:-3e-5}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.005}"
AGGREGATOR_TARGET_KL="${AGGREGATOR_TARGET_KL:-0.05}"
FULL_KL_COEF="${FULL_KL_COEF:-0.0}"
ENT_COEF="${ENT_COEF:-0.0}"
ENT_COEF_PHASE1="${ENT_COEF_PHASE1:-0.0}"
ENT_COEF_PHASE2="${ENT_COEF_PHASE2:-5e-4}"
ENT_COEF_PHASE3="${ENT_COEF_PHASE3:-1e-4}"
# The old defaults kept the run in aggregator-only / head-only phases for too
# long. The reference log under 20260630-153806 peaked around 0.2M-0.4M steps
# and degraded badly after ~1.1M while the backbone was still frozen.
PHASE1_END_STEP="${PHASE1_END_STEP:-300000}"
PHASE2_END_STEP="${PHASE2_END_STEP:-800000}"
BEST_SAVE_START_STEP="${BEST_SAVE_START_STEP:-}"
USE_VISION_LORA="${USE_VISION_LORA:-0}"
TRAIN_VISION_BACKBONE="${TRAIN_VISION_BACKBONE:-1}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
RESUME_DIR="${RESUME_DIR:-ckpt/TwoRobotPickCube-v2_ag/ppo/pandas_pandas/vla_adapter_smolvla_mappo/20260701-191720}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-1}"

FEATURE_SELECTOR_ALPHA="${FEATURE_SELECTOR_ALPHA:-0.2}"
FEATURE_SELECTOR_TOPK_TRAJECTORIES="${FEATURE_SELECTOR_TOPK_TRAJECTORIES:-4}"
FEATURE_SELECTOR_TEMPORAL_POOL_STEPS="${FEATURE_SELECTOR_TEMPORAL_POOL_STEPS:-16}"
FEATURE_SELECTOR_STRATEGY="${FEATURE_SELECTOR_STRATEGY:-return_span}"
EVAL_FEATURE_SELECTOR_STRATEGY="${EVAL_FEATURE_SELECTOR_STRATEGY:-}"

FEATURE_AGGREGATOR_ATTENTION_NUM_HEADS="${FEATURE_AGGREGATOR_ATTENTION_NUM_HEADS:-4}"
FEATURE_AGGREGATOR_GATE_TYPE="${FEATURE_AGGREGATOR_GATE_TYPE:-two-layers}"
FEATURE_AGGREGATOR_GATE_ACTIVATION="${FEATURE_AGGREGATOR_GATE_ACTIVATION:-relu}"
FEATURE_AGGREGATOR_NORM_TYPE="${FEATURE_AGGREGATOR_NORM_TYPE:-none}"
FEATURE_AGGREGATOR_FEATURE_GATE_OPEN_MAX="${FEATURE_AGGREGATOR_FEATURE_GATE_OPEN_MAX:-0.20}"
FEATURE_AGGREGATOR_ACTION_GATE_OPEN_MAX="${FEATURE_AGGREGATOR_ACTION_GATE_OPEN_MAX:-0.06}"
FEATURE_AGGREGATOR_Q_RET_WEIGHT="${FEATURE_AGGREGATOR_Q_RET_WEIGHT:-0.60}"
FEATURE_AGGREGATOR_Q_ATTN_WEIGHT="${FEATURE_AGGREGATOR_Q_ATTN_WEIGHT:-0.40}"
FEATURE_AGGREGATOR_REMOTE_DROPOUT_PROB="${FEATURE_AGGREGATOR_REMOTE_DROPOUT_PROB:-0.0}"
FEATURE_AGGREGATOR_REMOTE_NOISE_STD="${FEATURE_AGGREGATOR_REMOTE_NOISE_STD:-0.0}"
FEATURE_AGGREGATOR_REMOTE_STALE_SHIFT_MAX="${FEATURE_AGGREGATOR_REMOTE_STALE_SHIFT_MAX:-0}"

GATE_REG_COEF="${GATE_REG_COEF:-0.0}"
GATE_TARGET_MEAN="${GATE_TARGET_MEAN:-0.1}"
GATE_STD_COEF="${GATE_STD_COEF:-0.0}"
FEATURE_GATE_REG_COEF="${FEATURE_GATE_REG_COEF:-}"
FEATURE_GATE_TARGET_MEAN="${FEATURE_GATE_TARGET_MEAN:-0.1}"
FEATURE_GATE_STD_COEF="${FEATURE_GATE_STD_COEF:-}"
ACTION_GATE_REG_COEF="${ACTION_GATE_REG_COEF:-}"
ACTION_GATE_TARGET_MEAN="${ACTION_GATE_TARGET_MEAN:-0.03}"
ACTION_GATE_STD_COEF="${ACTION_GATE_STD_COEF:-}"
FEATURE_GATE_QUALITY_COEF="${FEATURE_GATE_QUALITY_COEF:-1.0}"
ACTION_GATE_QUALITY_COEF="${ACTION_GATE_QUALITY_COEF:-1.0}"
FEATURE_CONSISTENCY_COEF="${FEATURE_CONSISTENCY_COEF:-0.5}"
ACTION_CONSISTENCY_COEF="${ACTION_CONSISTENCY_COEF:-1.5}"
FEATURE_ATTN_ENTROPY_COEF="${FEATURE_ATTN_ENTROPY_COEF:-1e-4}"
ACTION_ATTN_ENTROPY_COEF="${ACTION_ATTN_ENTROPY_COEF:-1e-4}"
FEATURE_ATTN_DIVERSITY_COEF="${FEATURE_ATTN_DIVERSITY_COEF:-1e-4}"
ACTION_ATTN_DIVERSITY_COEF="${ACTION_ATTN_DIVERSITY_COEF:-1e-4}"
ENABLE_MIXED_TRAIN_ENVS="${ENABLE_MIXED_TRAIN_ENVS:-1}"
MIXED_ENV_CLEAN_RATIO="${MIXED_ENV_CLEAN_RATIO:-0.20}"
MIXED_ENV_MILD_RATIO="${MIXED_ENV_MILD_RATIO:-0.50}"
MIXED_ENV_HARD_RATIO="${MIXED_ENV_HARD_RATIO:-0.30}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
POLICY_MODE="${POLICY_MODE:-native}"
COMMUNICATION_REPLAY="${COMMUNICATION_REPLAY:-0}"

if [[ -z "${BEST_SAVE_START_STEP}" ]]; then
  BEST_SAVE_START_STEP="${PHASE1_END_STEP}"
fi

if [[ -z "${MAPPO_CKPT_PATH}" ]]; then
  echo "MAPPO_CKPT_PATH is empty. Set it to a MAPPO-pretrained checkpoint such as best_agent.pt or latest_agent.pt." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/mappo_feature_aggregator_pretrain.py
  --task-name TwoRobotPickCube-v2
  --ckpt-task-name TwoRobotPickCube-v2_ag
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --init-agent-path "${MAPPO_CKPT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --total-steps "${TOTAL_STEPS}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --critic-warmup-rollouts "${CRITIC_WARMUP_ROLLOUTS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --policy-mode "${POLICY_MODE}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --actor-hook-learning-rate "${ACTOR_HOOK_LEARNING_RATE}"
  --feature-aggregator-learning-rate "${FEATURE_AGGREGATOR_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --aggregator-target-kl "${AGGREGATOR_TARGET_KL}"
  --full-kl-coef "${FULL_KL_COEF}"
  --ent-coef "${ENT_COEF}"
  --ent-coef-phase1 "${ENT_COEF_PHASE1}"
  --ent-coef-phase2 "${ENT_COEF_PHASE2}"
  --ent-coef-phase3 "${ENT_COEF_PHASE3}"
  --best-save-start-step "${BEST_SAVE_START_STEP}"
  --phase1-end-step "${PHASE1_END_STEP}"
  --phase2-end-step "${PHASE2_END_STEP}"
  --feature-selector-alpha "${FEATURE_SELECTOR_ALPHA}"
  --feature-selector-strategy "${FEATURE_SELECTOR_STRATEGY}"
  --feature-aggregator-attention-num-heads "${FEATURE_AGGREGATOR_ATTENTION_NUM_HEADS}"
  --feature-aggregator-gate-type "${FEATURE_AGGREGATOR_GATE_TYPE}"
  --feature-aggregator-gate-activation "${FEATURE_AGGREGATOR_GATE_ACTIVATION}"
  --feature-aggregator-norm-type "${FEATURE_AGGREGATOR_NORM_TYPE}"
  --feature-aggregator-feature-gate-open-max "${FEATURE_AGGREGATOR_FEATURE_GATE_OPEN_MAX}"
  --feature-aggregator-action-gate-open-max "${FEATURE_AGGREGATOR_ACTION_GATE_OPEN_MAX}"
  --feature-aggregator-q-ret-weight "${FEATURE_AGGREGATOR_Q_RET_WEIGHT}"
  --feature-aggregator-q-attn-weight "${FEATURE_AGGREGATOR_Q_ATTN_WEIGHT}"
  --feature-aggregator-remote-dropout-prob "${FEATURE_AGGREGATOR_REMOTE_DROPOUT_PROB}"
  --feature-aggregator-remote-noise-std "${FEATURE_AGGREGATOR_REMOTE_NOISE_STD}"
  --feature-aggregator-remote-stale-shift-max "${FEATURE_AGGREGATOR_REMOTE_STALE_SHIFT_MAX}"
  --gate-reg-coef "${GATE_REG_COEF}"
  --gate-target-mean "${GATE_TARGET_MEAN}"
  --gate-std-coef "${GATE_STD_COEF}"
  --feature-gate-quality-coef "${FEATURE_GATE_QUALITY_COEF}"
  --action-gate-quality-coef "${ACTION_GATE_QUALITY_COEF}"
  --feature-consistency-coef "${FEATURE_CONSISTENCY_COEF}"
  --action-consistency-coef "${ACTION_CONSISTENCY_COEF}"
  --feature-attn-entropy-coef "${FEATURE_ATTN_ENTROPY_COEF}"
  --action-attn-entropy-coef "${ACTION_ATTN_ENTROPY_COEF}"
  --feature-attn-diversity-coef "${FEATURE_ATTN_DIVERSITY_COEF}"
  --action-attn-diversity-coef "${ACTION_ATTN_DIVERSITY_COEF}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
)

if [[ "${MODEL_BACKBONE}" == "openvla" ]]; then
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "MODEL_DIR is required when MODEL_BACKBONE=openvla." >&2
    exit 1
  fi
  CMD+=(--model-dir "${MODEL_DIR}")
  CMD+=(--use-vla-lora)
fi

if [[ "${USE_VISION_LORA}" == "1" ]]; then
  CMD+=(--use-vision-lora)
fi

if [[ "${TRAIN_VISION_BACKBONE}" == "1" ]]; then
  CMD+=(--train-vision-backbone)
fi

if [[ -n "${FEATURE_SELECTOR_TOPK_TRAJECTORIES}" ]]; then
  CMD+=(--feature-selector-topk-trajectories "${FEATURE_SELECTOR_TOPK_TRAJECTORIES}")
fi

if [[ -n "${FEATURE_SELECTOR_TEMPORAL_POOL_STEPS}" ]]; then
  CMD+=(--feature-selector-temporal-pool-steps "${FEATURE_SELECTOR_TEMPORAL_POOL_STEPS}")
fi

if [[ -n "${EVAL_FEATURE_SELECTOR_STRATEGY}" ]]; then
  CMD+=(--eval-feature-selector-strategy "${EVAL_FEATURE_SELECTOR_STRATEGY}")
fi

if [[ -n "${FEATURE_GATE_REG_COEF}" ]]; then
  CMD+=(--feature-gate-reg-coef "${FEATURE_GATE_REG_COEF}")
fi

if [[ -n "${FEATURE_GATE_TARGET_MEAN}" ]]; then
  CMD+=(--feature-gate-target-mean "${FEATURE_GATE_TARGET_MEAN}")
fi

if [[ -n "${FEATURE_GATE_STD_COEF}" ]]; then
  CMD+=(--feature-gate-std-coef "${FEATURE_GATE_STD_COEF}")
fi

if [[ -n "${ACTION_GATE_REG_COEF}" ]]; then
  CMD+=(--action-gate-reg-coef "${ACTION_GATE_REG_COEF}")
fi

if [[ -n "${ACTION_GATE_TARGET_MEAN}" ]]; then
  CMD+=(--action-gate-target-mean "${ACTION_GATE_TARGET_MEAN}")
fi

if [[ -n "${ACTION_GATE_STD_COEF}" ]]; then
  CMD+=(--action-gate-std-coef "${ACTION_GATE_STD_COEF}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${RESUME_USE_BEST_AGENT}" == "1" ]]; then
  CMD+=(--resume-use-best-agent)
fi

if [[ "${COMMUNICATION_REPLAY}" == "1" ]]; then
  CMD+=(--communication-replay)
else
  CMD+=(--no-communication-replay)
fi

if [[ "${ENABLE_MIXED_TRAIN_ENVS}" == "1" ]]; then
  CMD+=(
    --enable-mixed-train-envs
    --mixed-env-clean-ratio "${MIXED_ENV_CLEAN_RATIO}"
    --mixed-env-mild-ratio "${MIXED_ENV_MILD_RATIO}"
    --mixed-env-hard-ratio "${MIXED_ENV_HARD_RATIO}"
  )
fi

"${CMD[@]}"
