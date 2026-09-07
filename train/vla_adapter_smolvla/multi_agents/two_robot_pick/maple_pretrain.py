import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.internVL.maple_pretrain as maple_pretrain

from train.marl.maple import collect_rollout, maple_update_on_policy
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.maple_mixed_agent import (
    SmolVLAMapleAdapterAgent,
    build_optimizer,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import build_batch_from_obs


maple_pretrain.MODEL_NAME = "vla_adapter_smolvla_maple"
maple_pretrain.build_batch_from_obs = build_batch_from_obs
maple_pretrain.MapleEdgeVLAAgent = SmolVLAMapleAdapterAgent
maple_pretrain.build_optimizer = build_optimizer
maple_pretrain.collect_rollout = collect_rollout
maple_pretrain.maple_update_on_policy = maple_update_on_policy


def parse_args():
    argv = sys.argv[1:]
    cleaned_argv = []
    model_backbone = "mixed_tiny_vla_smolvla"
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

    if model_backbone != "mixed_tiny_vla_smolvla":
        raise ValueError(
            "SmolVLA MAPLE pretrain only supports --model-backbone mixed_tiny_vla_smolvla"
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
    return args


def main(args):
    args.model_backbone = "mixed_tiny_vla_smolvla"
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    for method_name in ("_adapt_actor_state_features", "get_action_and_value", "predict_future_state"):
        if method_name not in SmolVLAMapleAdapterAgent.__dict__:
            raise RuntimeError(
                f"SmolVLAMapleAdapterAgent is missing MAPLE override `{method_name}`; "
                "check train/vla_adapter_smolvla/multi_agents/two_robot_pick/maple_mixed_agent.py"
            )
    return maple_pretrain.main(args)


if __name__ == "__main__":
    main(parse_args())
