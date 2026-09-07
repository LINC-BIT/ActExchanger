#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MAPPO_CKPT_PATH="${MAPPO_CKPT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"
MODEL_BACKBONE="${MODEL_BACKBONE:-toy_cnn}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
TOTAL_STEPS="${TOTAL_STEPS:-2000000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}"
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
PHASE1_END_STEP="${PHASE1_END_STEP:-300000}"
PHASE2_END_STEP="${PHASE2_END_STEP:-800000}"
BEST_SAVE_START_STEP="${BEST_SAVE_START_STEP:-${PHASE1_END_STEP}}"
USE_VISION_LORA="${USE_VISION_LORA:-0}"
TRAIN_VISION_BACKBONE="${TRAIN_VISION_BACKBONE:-1}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"

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
MIX_TRAIN_ONLINE_ENVS="${MIX_TRAIN_ONLINE_ENVS:-1}"
ENABLE_MIXED_TRAIN_ENVS="${ENABLE_MIXED_TRAIN_ENVS:-0}"
USE_ONLINE_ENV_FAMILY_TRAIN_MIX="${USE_ONLINE_ENV_FAMILY_TRAIN_MIX:-1}"
TRAIN_ENV_ROLLOUTS_PER_FAMILY="${TRAIN_ENV_ROLLOUTS_PER_FAMILY:-4}"
ONLINE_ENV_FAMILY_TRAIN_MIX_MODE="${ONLINE_ENV_FAMILY_TRAIN_MIX_MODE:-random}"
ONLINE_ENV_FAMILY_CLEAN_WEIGHT="${ONLINE_ENV_FAMILY_CLEAN_WEIGHT:-0.20}"
ONLINE_ENV_FAMILY_VARIANT_WEIGHT="${ONLINE_ENV_FAMILY_VARIANT_WEIGHT:-0.16}"
MIXED_ENV_CLEAN_RATIO="${MIXED_ENV_CLEAN_RATIO:-0.20}"
MIXED_ENV_MILD_RATIO="${MIXED_ENV_MILD_RATIO:-0.50}"
MIXED_ENV_HARD_RATIO="${MIXED_ENV_HARD_RATIO:-0.30}"
MIXED_ENV_MILD_SIZE_MIN="${MIXED_ENV_MILD_SIZE_MIN:-0.70}"
MIXED_ENV_MILD_SIZE_MAX="${MIXED_ENV_MILD_SIZE_MAX:-1.25}"
MIXED_ENV_HARD_SIZE_MIN="${MIXED_ENV_HARD_SIZE_MIN:-0.50}"
MIXED_ENV_HARD_SIZE_MAX="${MIXED_ENV_HARD_SIZE_MAX:-0.70}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
POLICY_MODE="${POLICY_MODE:-native}"
COMMUNICATION_REPLAY="${COMMUNICATION_REPLAY:-0}"

if [[ -z "${MAPPO_CKPT_PATH}" ]]; then
  echo "MAPPO_CKPT_PATH is empty. Set it to a compatible mixed MAPPO checkpoint." >&2
  exit 1
fi

if [[ "${MIX_TRAIN_ONLINE_ENVS}" == "1" ]]; then
  USE_ONLINE_ENV_FAMILY_TRAIN_MIX="1"
  ENABLE_MIXED_TRAIN_ENVS="0"
fi

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/mappo_feature_aggregator_pretrain.py
  --task-name PassObjectFiveRobots-v1
  --ckpt-task-name PassObjectFiveRobots-v1_ag
  --robot-name panda_so100_widowx_xarm6_inspire
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --init-agent-path "${MAPPO_CKPT_PATH}"
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --max-episode-steps 300
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
  --feature-selector-topk-trajectories "${FEATURE_SELECTOR_TOPK_TRAJECTORIES}"
  --feature-selector-temporal-pool-steps "${FEATURE_SELECTOR_TEMPORAL_POOL_STEPS}"
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

if [[ "${USE_VISION_LORA}" == "1" ]]; then
  CMD+=(--use-vision-lora)
fi

if [[ "${TRAIN_VISION_BACKBONE}" == "1" ]]; then
  CMD+=(--train-vision-backbone)
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
    --mixed-env-mild-size-min "${MIXED_ENV_MILD_SIZE_MIN}"
    --mixed-env-mild-size-max "${MIXED_ENV_MILD_SIZE_MAX}"
    --mixed-env-hard-size-min "${MIXED_ENV_HARD_SIZE_MIN}"
    --mixed-env-hard-size-max "${MIXED_ENV_HARD_SIZE_MAX}"
  )
fi

if [[ "${USE_ONLINE_ENV_FAMILY_TRAIN_MIX}" == "1" ]]; then
  CMD+=(
    --use-online-env-family-train-mix
    --train-env-rollouts-per-family "${TRAIN_ENV_ROLLOUTS_PER_FAMILY}"
    --online-env-family-train-mix-mode "${ONLINE_ENV_FAMILY_TRAIN_MIX_MODE}"
    --online-env-family-clean-weight "${ONLINE_ENV_FAMILY_CLEAN_WEIGHT}"
    --online-env-family-variant-weight "${ONLINE_ENV_FAMILY_VARIANT_WEIGHT}"
  )
fi

"${CMD[@]}"
