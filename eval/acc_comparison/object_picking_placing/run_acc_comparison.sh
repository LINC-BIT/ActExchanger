#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpt}"

RUN_DIR="${RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_wo_ag/ppo/pandas_pandas/vla_adapter_smolvla_mappo/20260702-053209}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/ppo/pandas_pandas/vla_adapter_smolvla_mappo/20260630-183240}"
DICG_RUN_DIR="${DICG_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/dicg/pandas_pandas/vla_adapter_smolvla_dicg/20260630-183214}"
MAT_RUN_DIR="${MAT_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/mat/pandas_pandas/vla_adapter_smolvla_mat/20260630-183216}"
TGCNET_RUN_DIR="${TGCNET_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/tgcnet/pandas_pandas/vla_adapter_smolvla_tgcnet/20260630-183215}"
MAPLE_RUN_DIR="${MAPLE_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/maple/pandas_pandas/vla_adapter_smolvla_maple/20260701-073354}"
COMATRACK_RUN_DIR="${COMATRACK_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/comatrack/pandas_pandas/vla_adapter_smolvla_comatrack/20260701-093422}"
MAPORL_RUN_DIR="${MAPORL_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/maporl/pandas_pandas/vla_adapter_smolvla_maporl_planner/20260701-154416}"
MPDF_RUN_DIR="${MPDF_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/mpdf/pandas_pandas/vla_adapter_smolvla_mpdf_planner/20260701-154417}"
MAGRPO_RUN_DIR="${MAGRPO_RUN_DIR:-ckpt/TwoRobotPickCube-v2_online_baseline/magrpo/pandas_pandas/vla_adapter_smolvla_magrpo_planner/20260701-154247}"
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
  bash train/vla_adapter_smolvla/multi_agents/two_robot_pick/run_draw_online_rl_acc.sh \
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

This wrapper renders a 1x10 figure by default: one subplot per env, with
time on the x-axis and accuracy on the y-axis.

Required:
  --baseline-run-dir   baseline MAPPO run directory
  --run-dir            main run directory

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
  --swap-prob          swap probability, default: 0.8
  --nearby-radius      candidate radius, default: 1.0
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

if [[ -z "${BASELINE_RUN_DIR}" || -z "${RUN_DIR}" ]]; then
  echo "--baseline-run-dir and --run-dir are required." >&2
  usage >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/plot_acc_comparison.py"
  --baseline-run-dir "${BASELINE_RUN_DIR}"
  --run-dir "${RUN_DIR}"
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
