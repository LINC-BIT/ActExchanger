#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGO="${ALGO:-mat}"
INIT_AGENT_PATH="${INIT_AGENT_PATH:-ckpt/PassObjectFiveRobots-v1/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo_pretrain/20260712-131713/best_agent_full.pt}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-16}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
NUM_MINIBATCH="${NUM_MINIBATCH:-16}"
TOTAL_STEPS="${TOTAL_STEPS:-50000000}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-3e-6}"
HEAD_LEARNING_RATE="${HEAD_LEARNING_RATE:-1e-4}"
STATE_LEARNING_RATE="${STATE_LEARNING_RATE:-3e-6}"
VALUE_HEAD_LEARNING_RATE="${VALUE_HEAD_LEARNING_RATE:-1e-4}"
FUTURE_HEAD_LEARNING_RATE="${FUTURE_HEAD_LEARNING_RATE:-3e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
CLIP_EPS="${CLIP_EPS:-0.1}"
TARGET_KL="${TARGET_KL:-0.05}"
ENT_COEF="${ENT_COEF:-0.0}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_USE_BEST_AGENT="${RESUME_USE_BEST_AGENT:-0}"
VISION_TOKEN_POOL_SIZE="${VISION_TOKEN_POOL_SIZE:-16}"
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}"
FREEZE_VLA_BACKBONE="${FREEZE_VLA_BACKBONE:-0}"
MAGRPO_KL_COEF="${MAGRPO_KL_COEF:-}"
MAGRPO_GROUP_SIZE="${MAGRPO_GROUP_SIZE:-}"
MAGRPO_GAMMA="${MAGRPO_GAMMA:-}"
MAGRPO_ENT_COEF="${MAGRPO_ENT_COEF:-}"
MAPORL_TEAM_ADV_COEF="${MAPORL_TEAM_ADV_COEF:-}"
MAPORL_KL_COEF="${MAPORL_KL_COEF:-}"
MPDF_GROUP_SIZE="${MPDF_GROUP_SIZE:-}"
MPDF_RANK_TEMPERATURE="${MPDF_RANK_TEMPERATURE:-}"
MPDF_KL_COEF="${MPDF_KL_COEF:-}"
MPDF_ENT_COEF="${MPDF_ENT_COEF:-}"

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

case "${ALGO}" in
  mat|dicg|tgcnet|comatrack|magrpo|maporl|mpdf|maple)
    ENTRY="${PROJECT_ROOT}/train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/${ALGO}_pretrain.py"
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use one of: mat dicg tgcnet comatrack magrpo maporl mpdf maple" >&2
    exit 1
    ;;
esac

CMD=(
  "${PYTHON_BIN}" "${ENTRY}"
  --task-name PassObjectFiveRobots-v1
  --robot-name panda_so100_widowx_xarm6_inspire
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
  --critic-warmup-rollouts "${CRITIC_WARMUP_ROLLOUTS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatch "${NUM_MINIBATCH}"
  --use-amp
  --normalize-state
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}"
  --head-learning-rate "${HEAD_LEARNING_RATE}"
  --state-learning-rate "${STATE_LEARNING_RATE}"
  --value-head-learning-rate "${VALUE_HEAD_LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --clip-eps "${CLIP_EPS}"
  --target-kl "${TARGET_KL}"
  --ent-coef "${ENT_COEF}"
  --attn-implementation sdpa
)

if [[ "${ALGO}" != "maple" ]]; then
  CMD+=(
    --model-backbone toy_cnn
    --policy-mode native
    --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
    --full-kl-coef 0.0
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
else
  MAPLE_HEAD_LEARNING_RATE="${MAPLE_HEAD_LEARNING_RATE:-1e-5}"
  MAPLE_FUTURE_STATE_COEF="${MAPLE_FUTURE_STATE_COEF:-0.01}"
  MAPLE_LATENT_LAYERS="${MAPLE_LATENT_LAYERS:-2}"
  MAPLE_FUTURE_HORIZON="${MAPLE_FUTURE_HORIZON:-1}"
  MAPLE_RL_BC_COEF="${MAPLE_RL_BC_COEF:-0.0}"
  CMD+=(
    --model-backbone toy_cnn
    --policy-mode native
    --vision-token-pool-size "${VISION_TOKEN_POOL_SIZE}"
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
    --head-learning-rate "${MAPLE_HEAD_LEARNING_RATE}"
    --future-head-learning-rate "${FUTURE_HEAD_LEARNING_RATE}"
    --maple-future-state-coef "${MAPLE_FUTURE_STATE_COEF}"
    --maple-latent-layers "${MAPLE_LATENT_LAYERS}"
    --maple-future-horizon "${MAPLE_FUTURE_HORIZON}"
    --maple-rl-bc-coef "${MAPLE_RL_BC_COEF}"
  )
fi

if [[ -n "${INIT_AGENT_PATH}" ]]; then
  CMD+=(--init-agent-path "${INIT_AGENT_PATH}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${RESUME_USE_BEST_AGENT}" == "1" && "${ALGO}" != "maple" ]]; then
  CMD+=(--resume-use-best-agent)
fi

if [[ "${FREEZE_VLA_BACKBONE}" == "1" ]]; then
  CMD+=(--freeze-vla-backbone)
fi

if [[ -n "${MAGRPO_KL_COEF}" ]]; then
  CMD+=(--magrpo-kl-coef "${MAGRPO_KL_COEF}")
fi
if [[ -n "${MAGRPO_GROUP_SIZE}" ]]; then
  CMD+=(--magrpo-group-size "${MAGRPO_GROUP_SIZE}")
fi
if [[ -n "${MAGRPO_GAMMA}" ]]; then
  CMD+=(--magrpo-gamma "${MAGRPO_GAMMA}")
fi
if [[ -n "${MAGRPO_ENT_COEF}" ]]; then
  CMD+=(--magrpo-ent-coef "${MAGRPO_ENT_COEF}")
fi
if [[ -n "${MAPORL_TEAM_ADV_COEF}" ]]; then
  CMD+=(--maporl-team-adv-coef "${MAPORL_TEAM_ADV_COEF}")
fi
if [[ -n "${MAPORL_KL_COEF}" ]]; then
  CMD+=(--maporl-kl-coef "${MAPORL_KL_COEF}")
fi
if [[ -n "${MPDF_GROUP_SIZE}" ]]; then
  CMD+=(--mpdf-group-size "${MPDF_GROUP_SIZE}")
fi
if [[ -n "${MPDF_RANK_TEMPERATURE}" ]]; then
  CMD+=(--mpdf-rank-temperature "${MPDF_RANK_TEMPERATURE}")
fi
if [[ -n "${MPDF_KL_COEF}" ]]; then
  CMD+=(--mpdf-kl-coef "${MPDF_KL_COEF}")
fi
if [[ -n "${MPDF_ENT_COEF}" ]]; then
  CMD+=(--mpdf-ent-coef "${MPDF_ENT_COEF}")
fi

"${CMD[@]}"
