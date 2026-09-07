import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_online_rl as mappo_online_rl
from train.marl.comatrack.base import comatrack_update_on_policy

mappo_online_rl.UPDATE_FN = comatrack_update_on_policy
mappo_online_rl.ALGO_NAME = "comatrack"
mappo_online_rl.MODEL_NAME = "mixed_five_vla_pass_object_comatrack"

main = mappo_online_rl.main
parse_args = mappo_online_rl.parse_args


if __name__ == "__main__":
    main(parse_args())
