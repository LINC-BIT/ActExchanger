import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.marl.mappo.base import collect_rollout, mappo_update_on_policy
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mixed_agent import (
    MixedTinyVLAAdapterSmolVLAEfficientVLAAgent,
    build_mixed_mappo_optimizer,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.online_driver import (
    run_online_training,
)
import train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mappo_pretrain as mappo_pretrain


MODEL_NAME = "vla_adapter_smolvla_efficientvla_mappo"
ALGO_NAME = "ppo"
AGENT_CLS = MixedTinyVLAAdapterSmolVLAEfficientVLAAgent
OPTIMIZER_FN = build_mixed_mappo_optimizer
UPDATE_FN = mappo_update_on_policy
ROLLOUT_FN = collect_rollout


def build_agent(args, infos):
    return AGENT_CLS(
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
        tiny_vla_use_decode_cache=getattr(args, "tiny_vla_use_decode_cache", False),
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
    parser.add_argument("--feature-selector-alpha", type=float, default=0.2)
    parser.add_argument("--feature-selector-topk-trajectories", type=int, default=None)
    parser.add_argument("--feature-selector-temporal-pool-steps", type=int, default=None)
    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="topk_return",
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument(
        "--eval-feature-selector-strategy",
        type=str,
        default=None,
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument("--feature-aggregator-attention-num-heads", type=int, default=4)
    parser.add_argument(
        "--feature-aggregator-gate-type",
        type=str,
        default="two-layers",
        choices=["single-layer", "two-layers"],
    )
    parser.add_argument(
        "--feature-aggregator-gate-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh"],
    )
    parser.add_argument(
        "--feature-aggregator-norm-type",
        type=str,
        default="none",
        choices=["none", "layernorm"],
    )
    parser.add_argument("--feature-aggregator-feature-gate-open-max", type=float, default=0.20)
    parser.add_argument("--feature-aggregator-action-gate-open-max", type=float, default=0.06)
    parser.add_argument("--feature-aggregator-q-ret-weight", type=float, default=0.60)
    parser.add_argument("--feature-aggregator-q-attn-weight", type=float, default=0.40)
    parser.add_argument("--feature-aggregator-remote-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-stale-shift-max", type=int, default=0)
    parser.add_argument("--not-train-aggregator", action="store_true")
    parser.add_argument("--disable-ag-debug-histograms", action="store_true")
    extras, _ = parser.parse_known_args(sys.argv[1:])

    online_extra_flags = {
        "--ckpt-task-name",
        "--env-change-time-points",
        "--max-time",
        "--eval-episodes",
        "--feature-selector-alpha",
        "--feature-selector-topk-trajectories",
        "--feature-selector-temporal-pool-steps",
        "--feature-selector-strategy",
        "--eval-feature-selector-strategy",
        "--feature-aggregator-attention-num-heads",
        "--feature-aggregator-gate-type",
        "--feature-aggregator-gate-activation",
        "--feature-aggregator-norm-type",
        "--feature-aggregator-feature-gate-open-max",
        "--feature-aggregator-action-gate-open-max",
        "--feature-aggregator-q-ret-weight",
        "--feature-aggregator-q-attn-weight",
        "--feature-aggregator-remote-dropout-prob",
        "--feature-aggregator-remote-noise-std",
        "--feature-aggregator-remote-stale-shift-max",
    }
    online_extra_bool_flags = {"--not-train-aggregator", "--disable-ag-debug-histograms"}
    cleaned_argv = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in online_extra_bool_flags:
            continue
        if arg in online_extra_flags:
            skip_next = True
            continue
        cleaned_argv.append(arg)

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *cleaned_argv]
        args = mappo_pretrain.parse_args()
    finally:
        sys.argv = original_argv

    args.ckpt_task_name = extras.ckpt_task_name
    args.env_change_time_points = extras.env_change_time_points
    args.max_time = extras.max_time
    args.eval_episodes = extras.eval_episodes
    args.feature_selector_alpha = extras.feature_selector_alpha
    args.feature_selector_topk_trajectories = extras.feature_selector_topk_trajectories
    args.feature_selector_temporal_pool_steps = extras.feature_selector_temporal_pool_steps
    args.feature_selector_strategy = extras.feature_selector_strategy
    args.eval_feature_selector_strategy = extras.eval_feature_selector_strategy
    args.feature_aggregator_attention_num_heads = extras.feature_aggregator_attention_num_heads
    args.feature_aggregator_gate_type = extras.feature_aggregator_gate_type
    args.feature_aggregator_gate_activation = extras.feature_aggregator_gate_activation
    args.feature_aggregator_norm_type = extras.feature_aggregator_norm_type
    args.feature_aggregator_feature_gate_open_max = extras.feature_aggregator_feature_gate_open_max
    args.feature_aggregator_action_gate_open_max = extras.feature_aggregator_action_gate_open_max
    args.feature_aggregator_q_ret_weight = extras.feature_aggregator_q_ret_weight
    args.feature_aggregator_q_attn_weight = extras.feature_aggregator_q_attn_weight
    args.feature_aggregator_remote_dropout_prob = extras.feature_aggregator_remote_dropout_prob
    args.feature_aggregator_remote_noise_std = extras.feature_aggregator_remote_noise_std
    args.feature_aggregator_remote_stale_shift_max = extras.feature_aggregator_remote_stale_shift_max
    args.not_train_aggregator = extras.not_train_aggregator
    args.disable_ag_debug_histograms = extras.disable_ag_debug_histograms
    args.model_backbone = "mixed_tiny_vla_smolvla_efficientvla"
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return args


def main(args):
    args.model_backbone = "mixed_tiny_vla_smolvla_efficientvla"
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    return run_online_training(
        args,
        algo_name=ALGO_NAME,
        model_name=MODEL_NAME,
        build_agent=build_agent,
        build_optimizer=OPTIMIZER_FN,
        update_fn=UPDATE_FN,
        collect_rollout_fn=ROLLOUT_FN,
        rollout_mode="mappo",
        train_mode_during_update=False,
        init_label=f"{ALGO_NAME} online init",
    )


if __name__ == "__main__":
    main(parse_args())
