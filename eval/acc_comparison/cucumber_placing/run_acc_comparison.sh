#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpt}"
TASK_NAME="${TASK_NAME:-PlaceCucumber-v1_online_baseline}"
ROBOT_NAME="${ROBOT_NAME:-panda_widowx_widowx}"
MODEL_PREFIX="${MODEL_PREFIX:-vla_adapter_smolvla_efficientvla}"

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

RUN_DIR="${RUN_DIR-ckpt/PlaceCucumber-v1_online_wo_ag/ppo/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mappo/20260711-193017}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/ppo/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mappo/20260710-063926}"
DICG_RUN_DIR="${DICG_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/dicg/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_dicg/20260711-121607}"
MAT_RUN_DIR="${MAT_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/mat/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mat/20260711-124912}"
TGCNET_RUN_DIR="${TGCNET_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/tgcnet/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_tgcnet/20260711-124920}"
MAPLE_RUN_DIR="${MAPLE_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/maple/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maple/20260711-182233}"
COMATRACK_RUN_DIR="${COMATRACK_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/comatrack/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_comatrack/20260711-183439}"
MAPORL_RUN_DIR="${MAPORL_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/maporl/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maporl_planner/20260711-182919}"
MPDF_RUN_DIR="${MPDF_RUN_DIR-ckpt/PlaceCucumber-v1_online_baseline/mpdf/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mpdf_planner/20260711-183025}"
MAGRPO_RUN_DIR="${MAGRPO_RUN_DIR-}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/online_rl_acc_true.png}"
TAG="${TAG:-eval/success_once}"
TITLE="${TITLE:-Place Cucumber Online Success Curve}"
X_AXIS="${X_AXIS:-time}"
X_MAX="${X_MAX:-300}"
STRETCH_X="${STRETCH_X:-0}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-7}"
SEGMENT_TIME_POINTS="${SEGMENT_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"

usage() {
  cat <<'EOF'
Usage:
  bash train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber/run_draw_online_rl_acc_true.sh \
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
    [--title <plot_title>] \
    [--x-axis <time|step>] \
    [--x-max <max_x>] \
    [--stretch-x] \
    [--no-stretch-x]

Defaults:
  If a run directory is not specified, the script uses the newest timestamped run under:
  ckpt/PlaceCucumber-v1_online_baseline/<algo>/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_<algo>/

Required:
  --baseline-run-dir   baseline MAPPO run directory, default: newest MAPPO run
  --run-dir            main run directory, default: newest MAPPO run; pass empty to skip Ours

Optional:
  --dicg-run-dir       DICG run directory
  --mat-run-dir        MAT run directory
  --tgcnet-run-dir     TGCNet run directory
  --maple-run-dir      MAPLE run directory
  --comatrack-run-dir  CoMaTrack run directory
  --maporl-run-dir     MAPoRL run directory
  --mpdf-run-dir       MPDF run directory
  --magrpo-run-dir     MAGRPO run directory
  --output             output image path
  --tag                TensorBoard scalar tag, default: eval/success_once
  --title              plot title, default: Place Cucumber Online Success Curve
  --x-axis             time or step, default: time
  --x-max              x-axis max, default: 300
  --stretch-x          rescale curve x coordinates to x-max
  --no-stretch-x       do not rescale curve x coordinates to x-max
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
    --title)
      TITLE="$2"
      shift 2
      ;;
    --x-axis)
      X_AXIS="$2"
      shift 2
      ;;
    --x-max)
      X_MAX="$2"
      shift 2
      ;;
    --stretch-x)
      STRETCH_X="1"
      shift
      ;;
    --no-stretch-x)
      STRETCH_X="0"
      shift
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
  "${SCRIPT_DIR}/plot_acc_comparison.py"
  --baseline-run-dir "${BASELINE_RUN_DIR}"
  --output "${OUTPUT}"
  --tag "${TAG}"
  --title "${TITLE}"
  --x-axis "${X_AXIS}"
  --x-max "${X_MAX}"
  --smooth-window "${SMOOTH_WINDOW}"
  --segment-time-points "${SEGMENT_TIME_POINTS}"
)

if [[ "${STRETCH_X}" != "1" ]]; then
  CMD+=(--no-stretch-x)
fi

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
