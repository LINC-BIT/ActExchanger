import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs.pass_object_five_robots  # noqa: F401
import train.vla_adapter_smolvla.multi_agents.two_robot_pick.planner_pretrain as planner_pretrain

from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.reinforcement_learning.make_env import make_eval_envs
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_pretrain import (
    parse_args as parse_mappo_args,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.toy_cnn_agent import (
    PassObjectToyCNNAgent,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import (
    build_batch_from_obs,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.planner_model import (
    DEFAULT_SUBTASK_VOCAB,
    HighLevelSubtaskPlannerAgent,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.task_variants import (
    is_mixed_five_vla_backbone,
)


MODEL_NAME = "mixed_five_vla_pass_object_planner"
ALGO_NAME = "planner"
planner_update_on_policy = None
planner_build_rollout_meta = None


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


def build_low_level_agent(args, infos):
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    low_level_agent_path = getattr(args, "low_level_agent_path", None) or args.init_agent_path
    if low_level_agent_path is None:
        raise ValueError("--low-level-agent-path (or --init-agent-path for pretraining) is required")

    low_level_agent = PassObjectToyCNNAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
    )
    load_agent_checkpoint(low_level_agent, low_level_agent_path, map_location="cpu", label="planner low-level init")
    low_level_agent.freeze_state_stats()
    low_level_agent.eval()
    return low_level_agent


def parse_args():
    args = parse_mappo_args()
    args.ckpt_task_name = getattr(args, "ckpt_task_name", None)
    args.low_level_agent_path = getattr(args, "low_level_agent_path", None)
    args.planner_hidden_dim = getattr(args, "planner_hidden_dim", 256)
    args.planner_layers = getattr(args, "planner_layers", 2)
    args.planner_text_mode = getattr(args, "planner_text_mode", "fixed")
    args.planner_low_level_deterministic = getattr(args, "planner_low_level_deterministic", True)
    args.maporl_team_adv_coef = getattr(args, "maporl_team_adv_coef", 0.0)
    args.maporl_kl_coef = getattr(args, "maporl_kl_coef", 0.05)
    args.magrpo_eps = getattr(args, "magrpo_eps", 1e-5)
    args.magrpo_gamma = getattr(args, "magrpo_gamma", 1.0)
    args.magrpo_group_size = getattr(args, "magrpo_group_size", 0)
    args.magrpo_kl_coef = getattr(args, "magrpo_kl_coef", 0.05)
    args.magrpo_ent_coef = getattr(args, "magrpo_ent_coef", 0.0)
    args.mpdf_eps = getattr(args, "mpdf_eps", 1e-5)
    args.mpdf_group_size = getattr(args, "mpdf_group_size", 0)
    args.mpdf_rank_temperature = getattr(args, "mpdf_rank_temperature", 1.0)
    args.mpdf_kl_coef = getattr(args, "mpdf_kl_coef", 0.05)
    args.mpdf_ent_coef = getattr(args, "mpdf_ent_coef", 0.0)
    return args


def main(args):
    args.task_name = "PassObjectFiveRobots-v1"
    args.robot_name = "panda_so100_widowx_xarm6_inspire"
    planner_pretrain.MODEL_NAME = MODEL_NAME
    planner_pretrain.ALGO_NAME = ALGO_NAME
    planner_pretrain.planner_update_on_policy = planner_update_on_policy
    planner_pretrain.build_planner_rollout_meta = planner_build_rollout_meta
    planner_pretrain.build_batch_from_obs = build_batch_from_obs
    planner_pretrain.get_agent_info = get_agent_info
    planner_pretrain.build_low_level_agent = build_low_level_agent
    planner_pretrain.DEFAULT_SUBTASK_VOCAB = DEFAULT_SUBTASK_VOCAB
    planner_pretrain.HighLevelSubtaskPlannerAgent = HighLevelSubtaskPlannerAgent
    return planner_pretrain.main(args)


if __name__ == "__main__":
    main(parse_args())
