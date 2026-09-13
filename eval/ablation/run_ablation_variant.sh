#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  echo "Usage: bash eval/ablation/run_ablation_variant.sh <workload> <choice> <run_dir>" >&2
}

workload="${1:-}"
choice="${2:-}"
run_dir="${3:-}"
if [[ -z "${workload}" || -z "${choice}" || -z "${run_dir}" ]]; then
  usage
  exit 2
fi

case "${workload}" in
  object_picking_placing)
    method_dir="${REPO_ROOT}/train/vla_adapter_smolvla/multi_agents/two_robot_pick"
    ag_runner="${method_dir}/run_mappo_feature_aggregator_pretrain.sh"
    baseline_runner="${method_dir}/run_mappo_online_baseline.sh"
    ;;
  object_stacking)
    method_dir="${REPO_ROOT}/train/vla_adapter_openvla/multi_agents/two_robot_stack"
    ag_runner="${method_dir}/run_mappo_feature_aggregator_pretrain.sh"
    baseline_runner="${method_dir}/run_mappo_online_baseline.sh"
    ;;
  cucumber_placing)
    method_dir="${REPO_ROOT}/train/vla_adapter_smolvla_efficientvla/multi_agents/place_cucumber"
    ag_runner="${method_dir}/run_mappo_online_rl.sh"
    baseline_runner="${method_dir}/run_baseline_online.sh"
    ;;
  cylinder_transfer)
    method_dir="${REPO_ROOT}/train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots"
    ag_runner="${method_dir}/run_mappo_online_rl.sh"
    baseline_runner="${method_dir}/run_baseline_online.sh"
    ;;
  *)
    echo "Unknown workload: ${workload}" >&2
    exit 2
    ;;
esac

variant=""
temporal_pool_steps="${FEATURE_SELECTOR_TEMPORAL_POOL_STEPS:-}"
runner="${ag_runner}"
case "${choice}" in
  "Original length") variant="original_length" ;;
  "Raw features") variant="raw_features" ;;
  "Semantic-only features") variant="semantic_only_features" ;;
  "Spatial-only features") variant="spatial_only_features" ;;
  "Multiple unfused features per action") variant="multiple_unfused_features" ;;
  "No aggregation") runner="${baseline_runner}" ;;
  "Current-forward aggregation") variant="current_forward_aggregation" ;;
  "Random aggregation") variant="random_aggregation" ;;
  *)
    echo "Unsupported non-reference ablation choice: ${choice}" >&2
    exit 2
    ;;
esac

mkdir -p "${run_dir}"
env \
  ACTEXCHANGER_ABLATION_VARIANT="${variant}" \
  FEATURE_SELECTOR_TEMPORAL_POOL_STEPS="${temporal_pool_steps}" \
  RESUME_DIR="${run_dir}" \
  bash "${runner}"
