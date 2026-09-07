#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_ROOT="${CKPT_ROOT:-ckpt}"
TASK_NAME="${TASK_NAME:-PassObjectFiveRobots-v1_online_baseline}"
ROBOT_NAME="${ROBOT_NAME:-panda_so100_widowx_xarm6_inspire}"
MODEL_PREFIX="${MODEL_PREFIX:-mixed_five_vla_100m}"

latest_run_dir() {
  local algo="$1"
  local model="$2"
  local root="${CKPT_ROOT}/${TASK_NAME}/${algo}/${ROBOT_NAME}/${model}"
  local candidates=()
  shopt -s nullglob
  candidates=("${root}"/*)
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 0
  fi
  printf "%s\n" "${candidates[@]}" | sort | tail -n 1
}

RUN_DIR="${RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_wo_ag/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo/20260717-065701}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/ppo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mappo/20260716-103909}"
DICG_RUN_DIR="${DICG_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/dicg/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_dicg/20260716-105305}"
MAT_RUN_DIR="${MAT_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/mat/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mat/20260716-203542}"
TGCNET_RUN_DIR="${TGCNET_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/tgcnet/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_tgcnet/20260716-105301}"
MAPLE_RUN_DIR="${MAPLE_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/maple/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_maple/20260716-114605}"
COMATRACK_RUN_DIR="${COMATRACK_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/comatrack/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_comatrack/20260716-201112}"
MAPORL_RUN_DIR="${MAPORL_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/maporl/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_maporl_planner/20260717-083303}"
MPDF_RUN_DIR="${MPDF_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/mpdf/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_mpdf_planner/20260717-083304}"
MAGRPO_RUN_DIR="${MAGRPO_RUN_DIR-ckpt/PassObjectFiveRobots-v1_online_baseline/magrpo/panda_so100_widowx_xarm6_inspire/mixed_five_vla_pass_object_magrpo_planner/20260717-083301}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/online_rl_acc.png}"
TAG="${TAG:-eval/success_once}"
SWAP_PROB="${SWAP_PROB:-0.99}"
NEARBY_RADIUS="${NEARBY_RADIUS:-5.0}"
LAYOUT="${LAYOUT:-1x10}"
FONT_FAMILY="${FONT_FAMILY:-Arial}"
LABEL_FONTSIZE="${LABEL_FONTSIZE:-18}"
TITLE_FONTSIZE="${TITLE_FONTSIZE:-18}"
TICK_FONTSIZE="${TICK_FONTSIZE:-16}"

usage() {
  cat <<'EOF'
Usage:
  bash train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/run_draw_online_rl_acc.sh \
    --baseline-run-dir <baseline_run_dir> \
    --run-dir <run_dir> \
    [--dicg-run-dir <dicg_run_dir>] \
    [--mat-run-dir <mat_run_dir>] \
    [--tgcnet-run-dir <tgcnet_run_dir>] \
    [--maple-run-dir <maple_run_dir>] \
    [--comatrack-run-dir <comatrack_run_dir>] \
    [--maporl-run-dir <maporl_run_dir>] \
    [--mpdf-run-dir <mpdf_run_dir>] \
    [--magrpo-run-dir <magrpo_run_dir>] \
    [--output <output_png>] \
    [--tag <tb_scalar_tag>] \
    [--swap-prob <0-1>] \
    [--nearby-radius <float>] \
    [--layout <2x5|5x2|1x10>]

This wrapper renders a time-segment figure: one subplot per env,
with time on the x-axis and accuracy on the y-axis.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-run-dir)
      BASELINE_RUN_DIR="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --dicg-run-dir)
      DICG_RUN_DIR="$2"
      shift 2
      ;;
    --mat-run-dir)
      MAT_RUN_DIR="$2"
      shift 2
      ;;
    --tgcnet-run-dir)
      TGCNET_RUN_DIR="$2"
      shift 2
      ;;
    --maple-run-dir)
      MAPLE_RUN_DIR="$2"
      shift 2
      ;;
    --comatrack-run-dir)
      COMATRACK_RUN_DIR="$2"
      shift 2
      ;;
    --maporl-run-dir)
      MAPORL_RUN_DIR="$2"
      shift 2
      ;;
    --mpdf-run-dir)
      MPDF_RUN_DIR="$2"
      shift 2
      ;;
    --magrpo-run-dir)
      MAGRPO_RUN_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --swap-prob)
      SWAP_PROB="$2"
      shift 2
      ;;
    --nearby-radius)
      NEARBY_RADIUS="$2"
      shift 2
      ;;
    --layout)
      LAYOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${BASELINE_RUN_DIR}" ]]; then
  echo "Cannot resolve --baseline-run-dir automatically; pass it explicitly." >&2
  usage >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/draw_online_rl_acc.py"
  --baseline-run-dir "${BASELINE_RUN_DIR}"
  --output "${OUTPUT}"
  --tag "${TAG}"
  --swap-prob "${SWAP_PROB}"
  --nearby-radius "${NEARBY_RADIUS}"
  --layout "${LAYOUT}"
  --font-family "${FONT_FAMILY}"
  --label-fontsize "${LABEL_FONTSIZE}"
  --title-fontsize "${TITLE_FONTSIZE}"
  --tick-fontsize "${TICK_FONTSIZE}"
)

if [[ -n "${RUN_DIR}" ]]; then
  CMD+=(--run-dir "${RUN_DIR}")
fi
if [[ -n "${DICG_RUN_DIR}" ]]; then
  CMD+=(--dicg-run-dir "${DICG_RUN_DIR}")
fi
if [[ -n "${MAT_RUN_DIR}" ]]; then
  CMD+=(--mat-run-dir "${MAT_RUN_DIR}")
fi
if [[ -n "${TGCNET_RUN_DIR}" ]]; then
  CMD+=(--tgcnet-run-dir "${TGCNET_RUN_DIR}")
fi
if [[ -n "${MAPLE_RUN_DIR}" ]]; then
  CMD+=(--maple-run-dir "${MAPLE_RUN_DIR}")
fi
if [[ -n "${COMATRACK_RUN_DIR}" ]]; then
  CMD+=(--comatrack-run-dir "${COMATRACK_RUN_DIR}")
fi
if [[ -n "${MAPORL_RUN_DIR}" ]]; then
  CMD+=(--maporl-run-dir "${MAPORL_RUN_DIR}")
fi
if [[ -n "${MPDF_RUN_DIR}" ]]; then
  CMD+=(--mpdf-run-dir "${MPDF_RUN_DIR}")
fi
if [[ -n "${MAGRPO_RUN_DIR}" ]]; then
  CMD+=(--magrpo-run-dir "${MAGRPO_RUN_DIR}")
fi

"${CMD[@]}"
