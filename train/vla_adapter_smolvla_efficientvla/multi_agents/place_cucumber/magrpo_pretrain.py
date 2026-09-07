import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.planner_pretrain as planner_pretrain

from train.marl.magrpo.base import magrpo_update_on_policy


planner_pretrain.planner_update_on_policy = magrpo_update_on_policy
planner_pretrain.ALGO_NAME = "magrpo"
planner_pretrain.MODEL_NAME = "vla_adapter_smolvla_efficientvla_magrpo_planner"

main = planner_pretrain.main
parse_args = planner_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
