import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.marl.maple import collect_rollout, maple_update_on_policy
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.maple_mixed_agent import (
    ThreeVLAMapleAdapterAgent,
    build_optimizer,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.online_driver import (
    run_online_training,
)
import train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mappo_online_rl as mappo_online_rl


MODEL_NAME = "vla_adapter_smolvla_efficientvla_maple"
ALGO_NAME = "maple"


def build_agent(args, infos):
    return ThreeVLAMapleAdapterAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=args.freeze_vla_backbone,
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
        latent_layers=args.maple_latent_layers,
        future_horizon=args.maple_future_horizon,
    )


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--maple-latent-layers", type=int, default=2)
    parser.add_argument("--maple-future-horizon", type=int, default=1)
    parser.add_argument("--maple-future-state-coef", type=float, default=0.1)
    parser.add_argument("--maple-rl-bc-coef", type=float, default=0.0)
    maple_extras, _ = parser.parse_known_args(sys.argv[1:])

    maple_flags = {
        "--maple-latent-layers",
        "--maple-future-horizon",
        "--maple-future-state-coef",
        "--maple-rl-bc-coef",
    }
    cleaned_argv = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in maple_flags:
            skip_next = True
            continue
        cleaned_argv.append(arg)

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        args = mappo_online_rl.parse_args()
    finally:
        sys.argv = original_argv

    args.maple_latent_layers = maple_extras.maple_latent_layers
    args.maple_future_horizon = maple_extras.maple_future_horizon
    args.maple_future_state_coef = maple_extras.maple_future_state_coef
    args.maple_rl_bc_coef = maple_extras.maple_rl_bc_coef
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
        update_fn=maple_update_on_policy,
        collect_rollout_fn=collect_rollout,
        rollout_mode="maple",
        train_mode_during_update=True,
        init_label="MAPLE online init",
    )


if __name__ == "__main__":
    main(parse_args())
