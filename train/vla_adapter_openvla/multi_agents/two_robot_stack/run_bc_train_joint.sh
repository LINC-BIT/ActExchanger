#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "${PROJECT_ROOT}"
export USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
EXPERT_AGENT_DIR="${EXPERT_AGENT_DIR:-ckpt/TwoRobotStackCubeUR10e-v1/ppo/panda_ur10e_panda_gripper/toy_cnn/20260724-130231}"
TRAJECTORY_H5_PATH="${TRAJECTORY_H5_PATH:-}"
DATASET_CACHE_PATH="${DATASET_CACHE_PATH:-}"
COLLECT_ONLY="${COLLECT_ONLY:-0}"
FORCE_RECOLLECT_TRAJECTORIES="${FORCE_RECOLLECT_TRAJECTORIES:-0}"
FORCE_REBUILD_DATASET_CACHE="${FORCE_REBUILD_DATASET_CACHE:-0}"
VERIFY_REPLAY="${VERIFY_REPLAY:-1}"

RESUME_DIR="${RESUME_DIR:-ckpt/TwoRobotStackCubeUR10e-v1/sft/panda_ur10e_panda_gripper/vla_adapter_openvla_sft/20260726-180924}"

CMD=(
  "${PYTHON_BIN}" ${PROJECT_ROOT}/train/vla_adapter_openvla/multi_agents/two_robot_stack/bc_pretrain.py
  --task-name TwoRobotStackCubeUR10e-v1
  --model-backbone mixed_tiny_vla_openvla
  --image-size 112
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
  --entropy-coef "${ENTROPY_COEF}"
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

if [[ -n "${EXPERT_AGENT_DIR}" ]]; then
  CMD+=(--expert-agent-dir "${EXPERT_AGENT_DIR}")
elif [[ -z "${TRAJECTORY_H5_PATH}" ]]; then
  echo "Either EXPERT_AGENT_DIR or TRAJECTORY_H5_PATH must be set to obtain SFT data." >&2
  exit 1
fi

if [[ -n "${TRAJECTORY_H5_PATH}" ]]; then
  CMD+=(--trajectory-h5-path "${TRAJECTORY_H5_PATH}")
fi

if [[ -n "${DATASET_CACHE_PATH}" ]]; then
  CMD+=(--dataset-cache-path "${DATASET_CACHE_PATH}")
fi

if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume-dir "${RESUME_DIR}")
fi

if [[ "${COLLECT_ONLY}" == "1" ]]; then
  CMD+=(--collect-only)
fi

if [[ "${FORCE_RECOLLECT_TRAJECTORIES}" == "1" ]]; then
  CMD+=(--force-recollect-trajectories)
fi

if [[ "${FORCE_REBUILD_DATASET_CACHE}" == "1" ]]; then
  CMD+=(--force-rebuild-dataset-cache)
fi

if [[ "${VERIFY_REPLAY}" != "1" ]]; then
  CMD+=(--skip-replay-verify)
fi

"${CMD[@]}"
