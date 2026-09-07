#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_DIR="${RUN_DIR:-ckpt/PlaceCucumber-v1_online_wo_ag/ppo/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mappo/20260711-193017}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/ppo/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mappo/20260710-063926}"
DICG_RUN_DIR="${DICG_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/dicg/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_dicg/20260711-121607}"
MAT_RUN_DIR="${MAT_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/mat/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mat/20260711-124912}"
TGCNET_RUN_DIR="${TGCNET_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/tgcnet/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_tgcnet/20260711-124920}"
MAPLE_RUN_DIR="${MAPLE_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/maple/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maple/20260711-182233}"
COMATRACK_RUN_DIR="${COMATRACK_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/comatrack/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_comatrack/20260711-183439}"
MAPORL_RUN_DIR="${MAPORL_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/maporl/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maporl_planner/20260711-182919}"
MPDF_RUN_DIR="${MPDF_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/mpdf/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mpdf_planner/20260711-183025}"
MAGRPO_RUN_DIR="${MAGRPO_RUN_DIR:-ckpt/PlaceCucumber-v1_online_baseline/magrpo/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_magrpo_planner/20260712-070043}"
TAG="${TAG:-eval/success_once}"
WINDOW_INDEX="${WINDOW_INDEX:-0}"
AUTO_SEARCH_TARGET="${AUTO_SEARCH_TARGET:-0}"
SEARCH_START="${SEARCH_START:-}"
SEARCH_END="${SEARCH_END:-}"
MIN_OURS_TRAIN_TIME="${MIN_OURS_TRAIN_TIME:-1.0}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-ratio}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-7}"
TARGET_MODE="${TARGET_MODE:-topk_mean}"
TARGET_TOPK="${TARGET_TOPK:-3}"
EPS="${EPS:-0.01}"
MIN_CONSECUTIVE="${MIN_CONSECUTIVE:-2}"
SEGMENT_TIME_POINTS="${SEGMENT_TIME_POINTS:-[31,61,91,121,151,181,211,241,271,301]}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
MIN_THRESHOLD="${MIN_THRESHOLD:-0.0}"
MAX_THRESHOLD="${MAX_THRESHOLD:-1.0}"
UNREACHED_EXTRA_TIME="${UNREACHED_EXTRA_TIME:-1.0}"
CSV_OUTPUT="${CSV_OUTPUT:-}"
JSON_OUTPUT="${JSON_OUTPUT:-}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/measure_time_to_same_accuracy.py"
  --run-dir "${RUN_DIR}"
  --baseline-run-dir "${BASELINE_RUN_DIR}"
  --tag "${TAG}"
  --smooth-window "${SMOOTH_WINDOW}"
  --target-mode "${TARGET_MODE}"
  --target-topk "${TARGET_TOPK}"
  --eps "${EPS}"
  --min-consecutive "${MIN_CONSECUTIVE}"
  --segment-time-points "${SEGMENT_TIME_POINTS}"
)

if [[ "${AUTO_SEARCH_TARGET}" == "1" ]]; then
  CMD+=(--auto-search-target)
  CMD+=(--threshold-step "${THRESHOLD_STEP}")
  CMD+=(--min-threshold "${MIN_THRESHOLD}")
  CMD+=(--max-threshold "${MAX_THRESHOLD}")
  CMD+=(--unreached-extra-time "${UNREACHED_EXTRA_TIME}")
  CMD+=(--min-ours-train-time "${MIN_OURS_TRAIN_TIME}")
  CMD+=(--search-objective "${SEARCH_OBJECTIVE}")
  if [[ -n "${SEARCH_START}" ]]; then
    CMD+=(--search-start "${SEARCH_START}")
  fi
  if [[ -n "${SEARCH_END}" ]]; then
    CMD+=(--search-end "${SEARCH_END}")
  fi
else
  CMD+=(--window-index "${WINDOW_INDEX}")
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
if [[ -n "${CSV_OUTPUT}" ]]; then
  CMD+=(--csv "${CSV_OUTPUT}")
fi
if [[ -n "${JSON_OUTPUT}" ]]; then
  CMD+=(--json "${JSON_OUTPUT}")
fi

"${CMD[@]}"
