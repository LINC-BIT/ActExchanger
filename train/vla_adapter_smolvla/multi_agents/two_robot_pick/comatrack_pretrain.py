import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla.multi_agents.two_robot_pick.mappo_pretrain as mappo_pretrain

from train.marl.comatrack.base import comatrack_update_on_policy


# Reuse the existing SmolVLA MAPPO entrypoint and swap only the updater.
mappo_pretrain.mappo_update_on_policy = comatrack_update_on_policy
mappo_pretrain.ALGO_NAME = "comatrack"
mappo_pretrain.MODEL_NAME = "vla_adapter_smolvla_comatrack"

main = mappo_pretrain.main
parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
