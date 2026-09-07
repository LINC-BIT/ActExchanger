#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALGO=maple
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-ckpt/PlaceCucumber-v1/maple/panda_widowx_widowx/vla_adapter_smolvla_efficientvla_maple/20260711-172147}"
exec "${SCRIPT_DIR}/run_baseline_online.sh" "$@"
