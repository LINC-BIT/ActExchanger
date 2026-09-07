import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_online_rl as mappo_online_rl
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.baseline_adapters import (
    FiveVLADICGAdapterAgent,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.toy_cnn_agent import (
    build_toy_cnn_mappo_optimizer,
)


class FiveVLADICGOnlineAgent(FiveVLADICGAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = True

    def get_action_and_value(
        self,
        batch,
        actions_input=None,
        action_bins_input=None,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ):
        del action_bins_input
        return super().get_action_and_value(
            batch,
            actions_input=actions_input,
            deterministic=deterministic,
            return_token_logits=return_token_logits,
        )


def build_optimizer(args, agent):
    optimizer = build_toy_cnn_mappo_optimizer(args, agent)
    adapter_params = [p for p in agent.dicg_adapter.parameters() if p.requires_grad]
    if adapter_params:
        optimizer.add_param_group({"params": adapter_params, "lr": args.head_learning_rate, "group_name": "dicg_adapter"})
    return optimizer


mappo_online_rl.AGENT_CLS = FiveVLADICGOnlineAgent
mappo_online_rl.OPTIMIZER_FN = build_optimizer
mappo_online_rl.ALGO_NAME = "dicg"
mappo_online_rl.MODEL_NAME = "mixed_five_vla_pass_object_dicg"

parse_args = mappo_online_rl.parse_args


def main(args):
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    return mappo_online_rl.main(args)


if __name__ == "__main__":
    main(parse_args())
