import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_online_rl as mappo_online_rl
from train.marl.mat import mat_update_on_policy
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mat_pretrain import (
    ResidualMATPolicy,
    build_optimizer,
)

class ResidualMATOnlineAgent(ResidualMATPolicy):
    pass


mappo_online_rl.AGENT_CLS = ResidualMATOnlineAgent
mappo_online_rl.OPTIMIZER_FN = build_optimizer
mappo_online_rl.UPDATE_FN = mat_update_on_policy
mappo_online_rl.ALGO_NAME = "mat"
mappo_online_rl.MODEL_NAME = "mixed_five_vla_pass_object_mat"

parse_args = mappo_online_rl.parse_args


def main(args):
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    return mappo_online_rl.main(args)


if __name__ == "__main__":
    main(parse_args())
