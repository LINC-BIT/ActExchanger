import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_openvla.multi_agents.two_robot_stack.mappo_online_rl as mappo_online_rl
from train.marl.mat import mat_update_on_policy
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mat_pretrain import (
    OpenVLAMATAdapterAgent,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mixed_sft_agent import (
    build_mixed_mappo_optimizer,
)


class OpenVLAMATOnlineAgent(OpenVLAMATAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = True


def build_optimizer(args, agent):
    optimizer = build_mixed_mappo_optimizer(args, agent)
    adapter_params = [p for p in agent.mat_adapter.parameters() if p.requires_grad]
    if adapter_params:
        optimizer.add_param_group(
            {
                "params": adapter_params,
                "lr": args.head_learning_rate,
                "group_name": "mat_adapter",
            }
        )
    return optimizer


mappo_online_rl.AGENT_CLS = OpenVLAMATOnlineAgent
mappo_online_rl.OPTIMIZER_FN = build_optimizer
mappo_online_rl.UPDATE_FN = mat_update_on_policy
mappo_online_rl.ALGO_NAME = "mat"
mappo_online_rl.MODEL_NAME = "vla_adapter_openvla_mat"

main = mappo_online_rl.main
parse_args = mappo_online_rl.parse_args


SmolVLAMATOnlineAgent = OpenVLAMATOnlineAgent


if __name__ == "__main__":
    main(parse_args())
