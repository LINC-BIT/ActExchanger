import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs.pass_object_five_robots  # noqa: F401
import train.internVL.maple_pretrain as maple_pretrain

from train.marl.maple import collect_rollout, maple_update_on_policy
from train.reinforcement_learning.make_env import make_eval_envs
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.maple_mixed_agent import (
    FiveVLAMapleAdapterAgent,
    build_optimizer,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import (
    build_batch_from_obs,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.task_variants import (
    is_mixed_five_vla_backbone,
)


def get_agent_info(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = test_env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "state_dims": {name: batch[f"agent_states_{name}"].shape[-1] for name in agent_names},
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": test_env.single_action_space[agent_names[0]].shape[0],
        "action_dims": {name: test_env.single_action_space[name].shape[0] for name in agent_names},
    }
    test_env.close()
    return info


def parse_args():
    args = maple_pretrain.parse_args()
    args.task_name = "PassObjectFiveRobots-v1"
    args.robot_name = "panda_so100_widowx_xarm6_inspire"
    args.control_mode = "pd_joint_delta_pos"
    args.max_episode_steps = 300
    args.model_backbone = "toy_cnn"
    args.model_dir = None if getattr(args, "model_dir", None) == "" else args.model_dir
    if getattr(args, "model_dir", None) == "OpenGVLab/InternVL3_5-1B-Instruct":
        args.model_dir = None
    return args


def main(args):
    args.task_name = "PassObjectFiveRobots-v1"
    args.robot_name = "panda_so100_widowx_xarm6_inspire"
    args.control_mode = "pd_joint_delta_pos"
    args.max_episode_steps = 300
    args.model_backbone = "toy_cnn"
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return maple_pretrain.main(args)


maple_pretrain.MODEL_NAME = "mixed_five_vla_pass_object_maple"
maple_pretrain.build_batch_from_obs = build_batch_from_obs
maple_pretrain.get_agent_info = get_agent_info
maple_pretrain.MapleEdgeVLAAgent = FiveVLAMapleAdapterAgent
maple_pretrain.build_optimizer = build_optimizer
maple_pretrain.collect_rollout = collect_rollout
maple_pretrain.maple_update_on_policy = maple_update_on_policy


if __name__ == "__main__":
    main(parse_args())
