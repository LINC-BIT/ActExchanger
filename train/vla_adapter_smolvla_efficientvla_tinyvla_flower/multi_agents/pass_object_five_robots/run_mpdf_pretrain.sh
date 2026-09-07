#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPDF_GROUP_SIZE="${MPDF_GROUP_SIZE:-8}" \
MPDF_RANK_TEMPERATURE="${MPDF_RANK_TEMPERATURE:-0.5}" \
MPDF_KL_COEF="${MPDF_KL_COEF:-0.05}" \
MPDF_ENT_COEF="${MPDF_ENT_COEF:-0.0}" \
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}" \
ALGO=mpdf "${SCRIPT_DIR}/run_baseline_pretrain.sh"
