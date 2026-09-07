import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.marl.maporl.base import maporl_update_on_policy
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.planner_pretrain import (
    build_optimizer,
    collect_planner_rollout,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.online_driver import (
    run_online_training,
)
import train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.planner_pretrain as planner_pretrain
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.planner_model import (
    DEFAULT_SUBTASK_VOCAB,
    HighLevelSubtaskPlannerAgent,
)


MODEL_NAME = "vla_adapter_smolvla_efficientvla_maporl_planner"
ALGO_NAME = "maporl"


def build_agent(args, infos):
    low_level_agent = planner_pretrain.build_low_level_agent(args, infos)
    return HighLevelSubtaskPlannerAgent(
        low_level_agent=low_level_agent,
        agent_names=infos["agent_names"],
        state_dim=infos["state_dim"],
        global_state_dim=infos["global_state_dim"],
        action_dim=infos["action_dim"],
        state_dims=infos.get("state_dims"),
        normalize_state=args.normalize_state,
        planner_hidden_dim=args.planner_hidden_dim,
        planner_layers=args.planner_layers,
        planner_subtasks=DEFAULT_SUBTASK_VOCAB,
        planner_text_mode=args.planner_text_mode,
        low_level_deterministic=args.planner_low_level_deterministic,
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt-task-name", type=str, default=None)
    parser.add_argument("--low-level-agent-path", type=str, default=None)
    parser.add_argument(
        "--env-change-time-points",
        type=str,
        default="[31,61,91,121,151,181,211,241,271,301]",
    )
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=8)
    extras, _ = parser.parse_known_args(sys.argv[1:])

    online_extra_flags = {
        "--ckpt-task-name",
        "--low-level-agent-path",
        "--env-change-time-points",
        "--max-time",
        "--eval-episodes",
    }
    cleaned_argv = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in online_extra_flags:
            skip_next = True
            continue
        cleaned_argv.append(arg)

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        args = planner_pretrain.parse_args()
    finally:
        sys.argv = original_argv

    args.ckpt_task_name = extras.ckpt_task_name
    args.low_level_agent_path = extras.low_level_agent_path
    args.env_change_time_points = extras.env_change_time_points
    args.max_time = extras.max_time
    args.eval_episodes = extras.eval_episodes
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return args


def main(args):
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return run_online_training(
        args,
        algo_name=ALGO_NAME,
        model_name=MODEL_NAME,
        build_agent=build_agent,
        build_optimizer=build_optimizer,
        update_fn=maporl_update_on_policy,
        collect_rollout_fn=collect_planner_rollout,
        rollout_mode="mappo",
        train_mode_during_update=True,
        init_label=f"{ALGO_NAME} online planner init",
    )


if __name__ == "__main__":
    main(parse_args())
