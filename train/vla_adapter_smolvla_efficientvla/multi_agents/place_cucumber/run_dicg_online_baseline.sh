#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=dicg
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PlaceCucumber-v1/dicg/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_mappo/20260709-182409/best_agent.pt}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
