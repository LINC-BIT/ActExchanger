#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_BACKBONE="${MODEL_BACKBONE:-toy_cnn}"
MODEL_DIR="${MODEL_DIR:-}"
TRAJECTORY_H5_PATH="${TRAJECTORY_H5_PATH:-datasets/PassObjectFiveRobots-v1/rl/trajectory.rgb+state_dict.pd_joint_delta_pos.physx_cuda.h5}"
DATASET_CACHE_PATH="${DATASET_CACHE_PATH:-}"
RESUME_DIR="${RESUME_DIR:-}"
EXPERT_AGENT_PATH="${EXPERT_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"

SFT_TOTAL_ITERS="${SFT_TOTAL_ITERS:-}"
SFT_EPOCHS="${SFT_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_SUCCESSFUL_TRAJECTORIES="${NUM_SUCCESSFUL_TRAJECTORIES:-10240}"
NUM_COLLECT_ENVS="${NUM_COLLECT_ENVS:-1024}"
MAX_COLLECT_EPISODES="${MAX_COLLECT_EPISODES:-}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_INTERVAL_ITERS="${EVAL_INTERVAL_ITERS:-2000}"
LOG_INTERVAL_ITERS="${LOG_INTERVAL_ITERS:-2000}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-5}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-1e-4}"
ENTROPY_COEF="${ENTROPY_COEF:-1e-3}"
MIN_ENTROPY_PER_DIM="${MIN_ENTROPY_PER_DIM:-0.5}"
ENTROPY_FLOOR_COEF="${ENTROPY_FLOOR_COEF:-1e-2}"
USE_VISION_LORA="${USE_VISION_LORA:-0}"
TRAIN_VISION_BACKBONE="${TRAIN_VISION_BACKBONE:-0}"
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-0}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
SAVE_DATASET_CACHE="${SAVE_DATASET_CACHE:-0}"
FORCE_REBUILD_DATASET_CACHE="${FORCE_REBUILD_DATASET_CACHE:-0}"
FORCE_RECOLLECT_TRAJECTORIES="${FORCE_RECOLLECT_TRAJECTORIES:-0}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
EXPERT_SAMPLE_ACTIONS="${EXPERT_SAMPLE_ACTIONS:-0}"
EXPERT_NORMALIZE_STATE="${EXPERT_NORMALIZE_STATE:-1}"
COLLECT_SHADER_DIR="${COLLECT_SHADER_DIR:-minimal}"
REFIT_STATE_STATS="${REFIT_STATE_STATS:-0}"
RECORD_ENV_STATE="${RECORD_ENV_STATE:-0}"
REPLAY_FRIENDLY_COLLECT="${REPLAY_FRIENDLY_COLLECT:-0}"

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

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/bc_pretrain.py
  --task-name PassObjectFiveRobots-v1
  --robot-name panda_so100_widowx_xarm6_inspire
  --model-backbone "${MODEL_BACKBONE}"
  --image-size 112
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --max-episode-steps 300
  --num-successful-trajectories "${NUM_SUCCESSFUL_TRAJECTORIES}"
  --num-collect-envs "${NUM_COLLECT_ENVS}"
  --collect-shader-dir "${COLLECT_SHADER_DIR}"
  --sft-epochs "${SFT_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --eval-episodes "${EVAL_EPISODES}"
  --eval-interval-iters "${EVAL_INTERVAL_ITERS}"
  --log-interval-iters "${LOG_INTERVAL_ITERS}"
  --use-amp
  --normalize-state
  --policy-mode native
  --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --entropy-coef "${ENTROPY_COEF}"
  --entropy-floor-coef "${ENTROPY_FLOOR_COEF}"
  --attn-implementation sdpa
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

if [[ -n "${SFT_TOTAL_ITERS}" ]]; then
  CMD+=(--sft-total-iters "${SFT_TOTAL_ITERS}")
fi

if [[ -n "${MIN_ENTROPY_PER_DIM}" ]]; then
  CMD+=(--min-entropy-per-dim "${MIN_ENTROPY_PER_DIM}")
fi

if [[ -n "${TRAJECTORY_H5_PATH}" ]]; then
  CMD+=(--trajectory-h5-path "${TRAJECTORY_H5_PATH}")
fi

if [[ -n "${EXPERT_AGENT_PATH}" ]]; then
  CMD+=(--expert-agent-path "${EXPERT_AGENT_PATH}")
fi

if [[ -n "${DATASET_CACHE_PATH}" ]]; then
  CMD+=(--dataset-cache-path "${DATASET_CACHE_PATH}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ -n "${MODEL_DIR}" ]]; then
  CMD+=(--model-dir "${MODEL_DIR}")
fi

if [[ "${FREEZE_VLA_BACKBONE}" == "1" ]]; then
  CMD+=(--freeze-vla-backbone)
fi

if [[ "${USE_VISION_LORA}" == "1" ]]; then
  CMD+=(--use-vision-lora)
fi

if [[ "${TRAIN_VISION_BACKBONE}" == "1" ]]; then
  CMD+=(--train-vision-backbone)
fi

if [[ "${FORCE_REBUILD_DATASET_CACHE}" == "1" ]]; then
  CMD+=(--force-rebuild-dataset-cache)
fi

if [[ "${SAVE_DATASET_CACHE}" == "1" ]]; then
  CMD+=(--save-dataset-cache)
fi

if [[ "${FORCE_RECOLLECT_TRAJECTORIES}" == "1" ]]; then
  CMD+=(--force-recollect-trajectories)
fi

if [[ "${COLLECT_ONLY}" == "1" ]]; then
  CMD+=(--collect-only)
fi

if [[ "${EXPERT_SAMPLE_ACTIONS}" == "1" ]]; then
  CMD+=(--expert-sample-actions)
fi

if [[ "${EXPERT_NORMALIZE_STATE}" == "0" ]]; then
  CMD+=(--no-expert-normalize-state)
fi

if [[ -n "${MAX_COLLECT_EPISODES}" ]]; then
  CMD+=(--max-collect-episodes "${MAX_COLLECT_EPISODES}")
fi

if [[ "${REFIT_STATE_STATS}" == "1" ]]; then
  CMD+=(--refit-state-stats)
fi

if [[ "${RECORD_ENV_STATE}" == "1" ]]; then
  CMD+=(--record-env-state)
fi

if [[ "${REPLAY_FRIENDLY_COLLECT}" == "1" ]]; then
  CMD+=(--replay-friendly-collect)
fi

"${CMD[@]}"
