import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.internVL.maple_pretrain as maple_pretrain
import envs.two_robot_stack_cube_v1  # noqa: F401

from train.marl.maple import collect_rollout, maple_update_on_policy
from train.reinforcement_learning.make_env import make_eval_envs
from train.vla_adapter_openvla.multi_agents.two_robot_stack.maple_mixed_agent import (
    OpenVLAMapleAdapterAgent,
    build_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mixed_sft_agent import (
    CANONICAL_MIXED_MODEL_BACKBONE,
    is_mixed_model_backbone,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.model import build_batch_from_obs


maple_pretrain.MODEL_NAME = "vla_adapter_openvla_maple"
maple_pretrain.build_batch_from_obs = build_batch_from_obs
maple_pretrain.MapleEdgeVLAAgent = OpenVLAMapleAdapterAgent
maple_pretrain.build_optimizer = build_optimizer
maple_pretrain.collect_rollout = collect_rollout
maple_pretrain.maple_update_on_policy = maple_update_on_policy

SFT_COMPAT_CONFIG = {
    "policy_mode": "native",
    "vision_token_pool_size": 16,
    "tiny_hidden_dim": 640,
    "tiny_vision_layers": 7,
    "tiny_decoder_layers": 8,
    "tiny_attention_heads": 10,
    "tiny_patch_size": 14,
    "tiny_ffn_mult": 4,
    "tiny_num_action_bins": 256,
    "tiny_prompt_length": 24,
    "smolvla_hidden_dim": 384,
    "smolvla_vision_layers": 6,
    "smolvla_attention_heads": 6,
    "smolvla_patch_size": 14,
    "smolvla_ffn_mult": 4,
}


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
    action_dims = {
        agent_name: int(test_env.single_action_space[agent_name].shape[0])
        for agent_name in agent_names
    }
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": max(action_dims.values()),
        "action_dims": action_dims,
    }
    test_env.close()
    return info


maple_pretrain.get_agent_info = get_agent_info


def parse_args():
    argv = sys.argv[1:]
    cleaned_argv = []
    model_backbone = CANONICAL_MIXED_MODEL_BACKBONE
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--model-backbone":
            if idx + 1 >= len(argv):
                raise ValueError("--model-backbone expects a value")
            model_backbone = argv[idx + 1]
            idx += 2
            continue
        cleaned_argv.append(arg)
        idx += 1

    if not is_mixed_model_backbone(model_backbone):
        raise ValueError(
            f"OpenVLA MAPLE pretrain only supports --model-backbone {CANONICAL_MIXED_MODEL_BACKBONE} "
            "(legacy alias mixed_tiny_vla_smolvla is also accepted)"
        )

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        args = maple_pretrain.parse_args()
    finally:
        sys.argv = original_argv
    args.model_backbone = model_backbone
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    for key, value in SFT_COMPAT_CONFIG.items():
        setattr(args, key, value)
    return args


def main(args):
    args.model_backbone = CANONICAL_MIXED_MODEL_BACKBONE
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    for method_name in ("_adapt_actor_state_features", "get_action_and_value", "predict_future_state"):
        if method_name not in OpenVLAMapleAdapterAgent.__dict__:
            raise RuntimeError(
                f"OpenVLAMapleAdapterAgent is missing MAPLE override `{method_name}`; "
                "check train/vla_adapter_openvla/multi_agents/two_robot_stack/maple_mixed_agent.py"
            )
    return maple_pretrain.main(args)


if __name__ == "__main__":
    main(parse_args())
