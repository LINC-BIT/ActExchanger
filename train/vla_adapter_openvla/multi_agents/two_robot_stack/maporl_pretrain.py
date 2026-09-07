import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_openvla.multi_agents.two_robot_stack.planner_pretrain as planner_pretrain

from train.marl.maporl.base import maporl_update_on_policy


planner_pretrain.planner_update_on_policy = maporl_update_on_policy
planner_pretrain.ALGO_NAME = "maporl"
planner_pretrain.MODEL_NAME = "vla_adapter_openvla_maporl_planner"

main = planner_pretrain.main
parse_args = planner_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
