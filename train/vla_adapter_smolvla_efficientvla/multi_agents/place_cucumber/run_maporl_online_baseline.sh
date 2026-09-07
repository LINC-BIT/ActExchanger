#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=maporl
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PlaceCucumber-v1/maporl/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maporl_planner/20260711-063701}"
export LOW_LEVEL_AGENT_PATH="${LOW_LEVEL_AGENT_PATH:-ckpt/PlaceCucumber-v1/sft/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_sft/20260709-054512/best_agent.pt}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
