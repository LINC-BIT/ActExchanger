import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import envs.place_cucumber  # noqa: F401
import train.vla_adapter_smolvla.multi_agents.two_robot_pick.planner_pretrain as planner_pretrain

from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.reinforcement_learning.make_env import make_eval_envs
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mappo_pretrain import (
    parse_args as parse_mappo_args,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mixed_agent import (
    MixedTinyVLAAdapterSmolVLAEfficientVLAAgent,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.model import (
    build_batch_from_obs,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.planner_model import (
    DEFAULT_SUBTASK_VOCAB,
    HighLevelSubtaskPlannerAgent,
)


MODEL_NAME = "vla_adapter_smolvla_efficientvla_planner"
ALGO_NAME = "planner"
planner_update_on_policy = None


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
    }
    test_env.close()
    return info


def build_low_level_agent(args, infos):
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    low_level_agent_path = getattr(args, "low_level_agent_path", None) or args.init_agent_path
    if low_level_agent_path is None:
        raise ValueError("--low-level-agent-path (or --init-agent-path for pretraining) is required")

    low_level_agent = MixedTinyVLAAdapterSmolVLAEfficientVLAAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=True,
        critic_hidden_dim=args.critic_hidden_dim,
        attention_implementation=args.attn_implementation,
        image_size=args.image_size,
        use_vla_lora=args.use_vla_lora,
        use_vision_lora=args.use_vision_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_vision_backbone=args.train_vision_backbone,
        vision_token_pool_size=args.vision_token_pool_size,
        policy_mode=args.policy_mode,
        tiny_hidden_dim=args.tiny_hidden_dim,
        tiny_vision_layers=args.tiny_vision_layers,
        tiny_decoder_layers=args.tiny_decoder_layers,
        tiny_attention_heads=args.tiny_attention_heads,
        tiny_patch_size=args.tiny_patch_size,
        tiny_ffn_mult=args.tiny_ffn_mult,
        tiny_num_action_bins=args.tiny_num_action_bins,
        tiny_prompt_length=args.tiny_prompt_length,
        smolvla_hidden_dim=args.smolvla_hidden_dim,
        smolvla_vision_layers=args.smolvla_vision_layers,
        smolvla_attention_heads=args.smolvla_attention_heads,
        smolvla_patch_size=args.smolvla_patch_size,
        smolvla_ffn_mult=args.smolvla_ffn_mult,
        efficientvla_hidden_dim=args.efficientvla_hidden_dim,
        efficientvla_vision_layers=args.efficientvla_vision_layers,
        efficientvla_decoder_layers=args.efficientvla_decoder_layers,
        efficientvla_attention_heads=args.efficientvla_attention_heads,
        efficientvla_patch_size=args.efficientvla_patch_size,
        efficientvla_ffn_mult=args.efficientvla_ffn_mult,
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
    return args


def main(args):
    args.task_name = "PlaceCucumber-v1"
    args.robot_name = "panda_widowx_widowx"
    args.model_backbone = "mixed_tiny_vla_smolvla_efficientvla"
    args.critic_warmup_rollouts = 0
    planner_pretrain.MODEL_NAME = MODEL_NAME
    planner_pretrain.ALGO_NAME = ALGO_NAME
    planner_pretrain.planner_update_on_policy = planner_update_on_policy
    planner_pretrain.build_batch_from_obs = build_batch_from_obs
    planner_pretrain.get_agent_info = get_agent_info
    planner_pretrain.build_low_level_agent = build_low_level_agent
    planner_pretrain.DEFAULT_SUBTASK_VOCAB = DEFAULT_SUBTASK_VOCAB
    planner_pretrain.HighLevelSubtaskPlannerAgent = HighLevelSubtaskPlannerAgent
    return planner_pretrain.main(args)


if __name__ == "__main__":
    main(parse_args())
