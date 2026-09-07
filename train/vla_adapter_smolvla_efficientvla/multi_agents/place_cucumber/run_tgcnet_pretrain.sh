#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO=tgcnet "${SCRIPT_DIR}/run_baseline_pretrain.sh"
