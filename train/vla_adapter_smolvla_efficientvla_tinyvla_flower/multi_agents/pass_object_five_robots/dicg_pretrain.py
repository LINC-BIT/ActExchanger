import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_pretrain as mappo_pretrain

from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.baseline_adapters import (
    FiveVLADICGAdapterAgent,
    build_actor_only_optimizer,
)


def build_optimizer(args, agent):
    return build_actor_only_optimizer(args, agent)


def main(args):
    if args.resume_dir is None and not args.init_agent_path:
        raise ValueError("DICG pretrain requires --init-agent-path or --resume-dir")
    args.model_backbone = "toy_cnn"
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    return mappo_pretrain.main(args)


mappo_pretrain.MixedFiveVLA100MAgent = FiveVLADICGAdapterAgent
mappo_pretrain.MixedTinyVLAAdapterSmolVLAEfficientVLAAgent = FiveVLADICGAdapterAgent
mappo_pretrain.build_mixed_mappo_optimizer = build_optimizer
mappo_pretrain.ALGO_NAME = "dicg"
mappo_pretrain.MODEL_NAME = "mixed_five_vla_pass_object_dicg"

parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
