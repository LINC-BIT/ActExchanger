#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPORL_TEAM_ADV_COEF="${MAPORL_TEAM_ADV_COEF:-0.5}" \
MAPORL_KL_COEF="${MAPORL_KL_COEF:-0.05}" \
CRITIC_WARMUP_ROLLOUTS="${CRITIC_WARMUP_ROLLOUTS:-25}" \
ALGO=maporl "${SCRIPT_DIR}/run_baseline_pretrain.sh"
