import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from train.marl.maple import collect_rollout, maple_update_on_policy
import train.vla_adapter_openvla.multi_agents.two_robot_stack.maple_pretrain as maple_pretrain
from train.vla_adapter_openvla.multi_agents.two_robot_stack.maple_mixed_agent import (
    OpenVLAMapleOnlineAgent,
    build_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mixed_sft_agent import (
    CANONICAL_MIXED_MODEL_BACKBONE,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.online_driver import run_online_training


MODEL_NAME = "vla_adapter_openvla_maple"
ALGO_NAME = "maple"


def build_agent(args, infos):
    return OpenVLAMapleOnlineAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=args.freeze_vla_backbone,
        critic_hidden_dim=args.critic_hidden_dim,
        attention_implementation=args.attn_implementation,
        image_size=args.image_size,
        use_vla_lora=args.use_vla_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        latent_layers=args.maple_latent_layers,
        future_horizon=args.maple_future_horizon,
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt-task-name", type=str, default=None)
    parser.add_argument(
        "--env-change-time-points",
        type=str,
        default="[21,42,63,84,105,126,147,168,189,210]",
    )
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--model-backbone", type=str, default=CANONICAL_MIXED_MODEL_BACKBONE)
    parser.add_argument("--resume-use-best-agent", action="store_true")
    extras, _ = parser.parse_known_args(sys.argv[1:])

    argv = sys.argv[1:]
    cleaned_argv = []
    skip_next = False
    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if (
            arg == "--ckpt-task-name"
            or arg == "--env-change-time-points"
            or arg == "--max-time"
            or arg == "--model-backbone"
            or arg == "--resume-use-best-agent"
        ):
            if arg != "--resume-use-best-agent":
                skip_next = True
            continue
        cleaned_argv.append(arg)

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        args = maple_pretrain.parse_args()
    finally:
        sys.argv = original_argv

    args.ckpt_task_name = extras.ckpt_task_name
    args.env_change_time_points = extras.env_change_time_points
    args.max_time = extras.max_time
    args.eval_episodes = extras.eval_episodes
    args.resume_use_best_agent = extras.resume_use_best_agent
    args.model_backbone = CANONICAL_MIXED_MODEL_BACKBONE
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return args


def main(args):
    args.model_backbone = CANONICAL_MIXED_MODEL_BACKBONE
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return run_online_training(
        args,
        algo_name=ALGO_NAME,
        model_name=MODEL_NAME,
        build_agent=build_agent,
        build_optimizer=build_optimizer,
        update_fn=maple_update_on_policy,
        collect_rollout_fn=collect_rollout,
        rollout_mode="maple",
        train_mode_during_update=True,
        init_label="MAPLE online init",
    )


if __name__ == "__main__":
    main(parse_args())
