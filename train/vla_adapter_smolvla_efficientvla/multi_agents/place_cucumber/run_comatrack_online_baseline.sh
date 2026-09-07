#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=comatrack
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PlaceCucumber-v1/comatrack/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_comatrack/20260711-180248}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
