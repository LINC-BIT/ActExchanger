import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mappo_pretrain as mappo_pretrain

from train.marl.mat import mat_update_on_policy
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.baseline_adapters import (
    ThreeVLAMATAdapterAgent,
    build_actor_only_optimizer,
)


def build_optimizer(args, agent):
    return build_actor_only_optimizer(args, agent)


def main(args):
    if args.resume_dir is None and not args.init_agent_path:
        raise ValueError("MAT pretrain requires --init-agent-path or --resume-dir")
    args.model_backbone = "mixed_tiny_vla_smolvla_efficientvla"
    args.critic_warmup_rollouts = 0
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    return mappo_pretrain.main(args)


mappo_pretrain.MixedTinyVLAAdapterSmolVLAEfficientVLAAgent = ThreeVLAMATAdapterAgent
mappo_pretrain.build_mixed_mappo_optimizer = build_optimizer
mappo_pretrain.mappo_update_on_policy = mat_update_on_policy
mappo_pretrain.ALGO_NAME = "mat"
mappo_pretrain.MODEL_NAME = "vla_adapter_smolvla_efficientvla_mat"

parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
