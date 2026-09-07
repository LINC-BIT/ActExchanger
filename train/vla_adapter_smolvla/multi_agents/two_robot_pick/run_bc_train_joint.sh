#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-64}"

RESUME_DIR="${RESUME_DIR:-}"

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick/bc_pretrain.py
  --task-name TwoRobotPickCube-v2
  --model-backbone mixed_tiny_vla_smolvla
  --image-size 112
  --expert-agent-dir ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942
  --obs-mode rgb+state_dict
  --control-mode pd_joint_delta_pos
  --expert-control-mode pd_joint_delta_pos
  --reward-mode normalized_dense
  --num-successful-trajectories 10240
  --num-collect-envs 1024
  --sft-total-iters 800000
  --batch-size "${BATCH_SIZE}"
  --log-interval-iters 2000
  --state-stats-batch-size 1024
  --eval-interval-iters 2000
  --eval-episodes 50
  --num-eval-envs 16
  --normalize-state
  --use-amp
  --policy-mode native
  --vision-token-pool-size 16
  --backbone-learning-rate 1e-4
  --head-learning-rate 2e-4
  --state-learning-rate 2e-4
  --attn-implementation sdpa
  --tiny-hidden-dim 640
  --tiny-vision-layers 7
  --tiny-decoder-layers 8
  --tiny-attention-heads 10
  --tiny-patch-size 14
  --tiny-ffn-mult 4
  --tiny-num-action-bins 256
  --tiny-prompt-length 24
)

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

"${CMD[@]}"
