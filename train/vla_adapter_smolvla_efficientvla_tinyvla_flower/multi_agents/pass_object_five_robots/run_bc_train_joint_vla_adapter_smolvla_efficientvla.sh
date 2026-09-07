#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_BACKBONE="${MODEL_BACKBONE:-toy_cnn}"
export SFT_TOTAL_ITERS="${SFT_TOTAL_ITERS:-800000}"
export TINY_PROMPT_LENGTH="${TINY_PROMPT_LENGTH:-24}"
exec "${SCRIPT_DIR}/run_bc_pretrain.sh" "$@"
