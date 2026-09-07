#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1_ag/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo_pretrain/20260716-190713/best_agent.pt}"
MODEL_DIR="${MODEL_DIR:-}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-50}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
TOTAL_STEPS="${TOTAL_STEPS:-50000000}"
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
TARGET_KL="${TARGET_KL:-0.05}"
ENT_COEF="${ENT_COEF:-0.0}"
FULL_KL_COEF="${FULL_KL_COEF:-0.0}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
MAX_TIME="${MAX_TIME:-}"
ENV_CHANGE_TIME_POINTS="${ENV_CHANGE_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
POLICY_MODE="${POLICY_MODE:-native}"

TINY_HIDDEN_DIM="${TINY_HIDDEN_DIM:-384}"
TINY_VISION_LAYERS="${TINY_VISION_LAYERS:-4}"
TINY_DECODER_LAYERS="${TINY_DECODER_LAYERS:-4}"
TINY_ATTENTION_HEADS="${TINY_ATTENTION_HEADS:-6}"
TINY_PATCH_SIZE="${TINY_PATCH_SIZE:-14}"
TINY_FFN_MULT="${TINY_FFN_MULT:-4}"
TINY_NUM_ACTION_BINS="${TINY_NUM_ACTION_BINS:-256}"
TINY_PROMPT_LENGTH="${TINY_PROMPT_LENGTH:-24}"
TINY_VLA_USE_DECODE_CACHE="${TINY_VLA_USE_DECODE_CACHE:-1}"
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
FEATURE_SELECTOR_STRATEGY="${FEATURE_SELECTOR_STRATEGY:-topk_return}"
EVAL_FEATURE_SELECTOR_STRATEGY="${EVAL_FEATURE_SELECTOR_STRATEGY:-}"
FEATURE_AGGREGATOR_ATTENTION_NUM_HEADS="${FEATURE_AGGREGATOR_ATTENTION_NUM_HEADS:-4}"
FEATURE_AGGREGATOR_GATE_TYPE="${FEATURE_AGGREGATOR_GATE_TYPE:-two-layers}"
FEATURE_AGGREGATOR_GATE_ACTIVATION="${FEATURE_AGGREGATOR_GATE_ACTIVATION:-relu}"
FEATURE_AGGREGATOR_NORM_TYPE="${FEATURE_AGGREGATOR_NORM_TYPE:-none}"
FEATURE_AGGREGATOR_FEATURE_GATE_OPEN_MAX="${FEATURE_AGGREGATOR_FEATURE_GATE_OPEN_MAX:-0.25}"
FEATURE_AGGREGATOR_ACTION_GATE_OPEN_MAX="${FEATURE_AGGREGATOR_ACTION_GATE_OPEN_MAX:-0.10}"
FEATURE_AGGREGATOR_Q_RET_WEIGHT="${FEATURE_AGGREGATOR_Q_RET_WEIGHT:-0.85}"
FEATURE_AGGREGATOR_Q_ATTN_WEIGHT="${FEATURE_AGGREGATOR_Q_ATTN_WEIGHT:-0.15}"
FEATURE_AGGREGATOR_REMOTE_DROPOUT_PROB="${FEATURE_AGGREGATOR_REMOTE_DROPOUT_PROB:-0.0}"
FEATURE_AGGREGATOR_REMOTE_NOISE_STD="${FEATURE_AGGREGATOR_REMOTE_NOISE_STD:-0.0}"
FEATURE_AGGREGATOR_REMOTE_STALE_SHIFT_MAX="${FEATURE_AGGREGATOR_REMOTE_STALE_SHIFT_MAX:-0}"
DISABLE_AG_DEBUG_HISTOGRAMS="${DISABLE_AG_DEBUG_HISTOGRAMS:-0}"

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/mappo_online_rl.py
  --task-name PassObjectFiveRobots-v1
  --robot-name panda_so100_widowx_xarm6_inspire
  --model-backbone toy_cnn
  --image-size 112
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --max-episode-steps 300
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
  --policy-mode "${POLICY_MODE}"
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --ent-coef "${ENT_COEF}"
  --full-kl-coef "${FULL_KL_COEF}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --env-change-time-points "${ENV_CHANGE_TIME_POINTS}"
  --not-train-aggregator
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
)

if [[ "${TINY_VLA_USE_DECODE_CACHE}" == "1" ]]; then
  CMD+=(--tiny-vla-use-decode-cache)
fi
if [[ -n "${EVAL_FEATURE_SELECTOR_STRATEGY}" ]]; then
  CMD+=(--eval-feature-selector-strategy "${EVAL_FEATURE_SELECTOR_STRATEGY}")
fi
if [[ "${DISABLE_AG_DEBUG_HISTOGRAMS}" == "1" ]]; then
  CMD+=(--disable-ag-debug-histograms)
fi

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
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

"${CMD[@]}"
