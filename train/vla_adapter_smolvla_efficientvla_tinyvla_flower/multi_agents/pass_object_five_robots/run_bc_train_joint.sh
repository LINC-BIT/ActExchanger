#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SFT_TOTAL_ITERS="${SFT_TOTAL_ITERS:-800000}"
exec "${SCRIPT_DIR}/run_bc_pretrain.sh" "$@"
