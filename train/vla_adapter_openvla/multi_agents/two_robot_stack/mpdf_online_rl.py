import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_openvla.multi_agents.two_robot_stack.planner_online_rl as planner_online_rl
from train.marl.mpdf.base import mpdf_update_on_policy


planner_online_rl.planner_update_on_policy = mpdf_update_on_policy
planner_online_rl.ALGO_NAME = "mpdf"
planner_online_rl.MODEL_NAME = "vla_adapter_openvla_mpdf_planner"

main = planner_online_rl.main
parse_args = planner_online_rl.parse_args


if __name__ == "__main__":
    main(parse_args())
