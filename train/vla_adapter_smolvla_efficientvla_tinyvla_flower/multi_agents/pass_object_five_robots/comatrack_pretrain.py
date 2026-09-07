import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_pretrain as mappo_pretrain

from train.marl.comatrack.base import comatrack_update_on_policy


mappo_pretrain.UPDATE_FN = comatrack_update_on_policy
mappo_pretrain.ALGO_NAME = "comatrack"
mappo_pretrain.MODEL_NAME = "mixed_five_vla_pass_object_comatrack"

main = mappo_pretrain.main
parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
