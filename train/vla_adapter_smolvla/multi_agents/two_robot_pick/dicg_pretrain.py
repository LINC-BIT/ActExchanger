import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla.multi_agents.two_robot_pick.mappo_pretrain as mappo_pretrain

from train.vla_adapter_smolvla.multi_agents.two_robot_pick.frozen_backbone_adapters import (
    DICGStateAdapter,
    FrozenMixedBackboneAdapterAgent,
    PerAgentProjectInOutAdapter,
    build_actor_only_optimizer,
)


class SmolVLADICGAdapterAgent(FrozenMixedBackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dicg_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.vla_actor.hidden_dim, self.smolvla_actor.hidden_dim],
            shared_dim=256,
            core=DICGStateAdapter(hidden_dim=256),
        )

    def _mixed_residuals(self, vla_state_feature, smolvla_state_feature):
        delta_vla, delta_smol = self.dicg_adapter([vla_state_feature, smolvla_state_feature])
        return delta_vla, delta_smol


def build_optimizer(args, agent):
    return build_actor_only_optimizer(args, agent)


def main(args):
    if args.resume_dir is None and not args.init_agent_path:
        raise ValueError("DICG pretrain requires --init-agent-path or --resume-dir")
    args.model_backbone = "mixed_tiny_vla_smolvla"
    args.critic_warmup_rollouts = 0
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    mappo_pretrain.main(args)


mappo_pretrain.MixedTinyVLAAdapterSmolVLASFTAgent = SmolVLADICGAdapterAgent
mappo_pretrain.build_mixed_mappo_optimizer = build_optimizer
mappo_pretrain.ALGO_NAME = "dicg"
mappo_pretrain.MODEL_NAME = "vla_adapter_smolvla_dicg"

parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
