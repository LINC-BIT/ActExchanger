#!/usr/bin/env python3
"""Benchmark ActExchanger's feature selector and feature aggregator."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ours.de_feature_fusion.feature_aggregator import FeatureAggregator
from ours.de_feature_fusion.vla_feature_aggregator import (
    VLAActionPositionFeatureSelector,
    VLAClientForMultiAgent,
)


WORKLOADS = {
    "object_picking_placing": ("Object picking and placing", 2),
    "object_stacking": ("Object stacking", 2),
    "cucumber_placing": ("Cucumber placing", 2),
    "cylinder_transfer": ("Cylinder transfer", 5),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device: torch.device, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    values = []
    for _ in range(samples):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return values


class ToyVLA(torch.nn.Module):
    def __init__(self, positions: int):
        super().__init__()
        self.action_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(positions))
        self.actor_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(positions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for feature_block, actor_block in zip(self.action_blocks, self.actor_blocks):
            features = actor_block(feature_block(features))
        return features


def build_selector(args: argparse.Namespace, model: ToyVLA) -> VLAActionPositionFeatureSelector:
    return VLAActionPositionFeatureSelector(
        model,
        "action_blocks",
        num_action_positions=args.positions,
        alpha=args.selector_alpha,
        max_trajectory_count=args.selector_topk_trajectories,
        temporal_pool_steps=args.selector_temporal_pool_steps,
        selection_strategy="topk_return",
        max_episode_steps=args.steps,
    )


def run_selector_rollout(selector, model, rollout_features, actions) -> dict:
    selector.reset_cache()
    trajectories = rollout_features.shape[1]
    for step, features in enumerate(rollout_features):
        model(features)
        selector.cache_features_of_high_reward_after_each_forward_during_rollout(
            rewards=torch.full((trajectories,), float(step + 1)),
            dones=torch.full((trajectories,), step == len(rollout_features) - 1),
            action_mean=actions[step],
            success=torch.arange(trajectories).remainder(2).eq(0),
            action_match=torch.ones(trajectories, dtype=torch.bool),
        )
    message = selector.select_message()
    if message is None:
        raise RuntimeError("Feature selector did not produce a message")
    return message


def build_receiver(args: argparse.Namespace, device: torch.device, peers: int):
    model = ToyVLA(args.positions).to(device)
    layer_names = [f"action_blocks.{index}" for index in range(args.positions)]
    actor_layer_names = [f"actor_blocks.{index}" for index in range(args.positions)]
    receiver = VLAClientForMultiAgent.__new__(VLAClientForMultiAgent)
    receiver.num_action_positions = args.positions
    receiver.feature_aggregators = {}
    for peer_index in range(peers):
        aggregator = FeatureAggregator(
            model,
            layer_names[0],
            args.feature_dim,
            args.feature_dim,
            remote_action_dim=args.action_dim,
            action_position_layer_names=layer_names,
            action_position_actor_layer_names=actor_layer_names,
            attention_num_heads=args.attention_heads,
            gate_type=args.gate_type,
        )
        aggregator.module.to(device).eval()
        receiver.feature_aggregators[f"agent_{peer_index}"] = aggregator
    return model, receiver


def benchmark_one(args: argparse.Namespace, workload_id: str) -> dict:
    workload, num_agents = WORKLOADS[workload_id]
    device = torch.device(args.device)
    selector_model = ToyVLA(args.positions)
    selector = build_selector(args, selector_model)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rollout_features = torch.randn(
        args.steps, args.trajectories, args.feature_dim, generator=generator
    )
    rollout_actions = torch.randn(
        args.steps, args.trajectories, args.action_dim, generator=generator
    )

    def select_features():
        return run_selector_rollout(selector, selector_model, rollout_features, rollout_actions)

    module_1_times = measure(select_features, device, args.warmup, args.samples)
    peer_messages = [select_features() for _ in range(num_agents - 1)]
    receiver_model, receiver = build_receiver(args, device, num_agents - 1)
    local = torch.randn(
        args.batch_size, args.feature_dim, device=device,
        generator=torch.Generator(device=device).manual_seed(args.seed + 2),
    )

    def aggregate():
        for peer_index, message in enumerate(peer_messages):
            VLAClientForMultiAgent.receive_feature_and_action(
                receiver, f"agent_{peer_index}", message
            )
        return receiver_model(local)

    with torch.inference_mode():
        module_2_times = measure(aggregate, device, args.warmup, args.samples)
    return {
        "workload": workload,
        "workload_id": workload_id,
        "num_agents": num_agents,
        "module_1_name": "Feature selector (cache + export)",
        "module_2_name": "Feature aggregator (all peers)",
        "module_1_sample_ms": statistics.median(module_1_times),
        "module_1_mean_ms": statistics.mean(module_1_times),
        "module_2_sample_ms": statistics.median(module_2_times),
        "module_2_mean_ms": statistics.mean(module_2_times),
        "module_1_samples_ms": module_1_times,
        "module_2_samples_ms": module_2_times,
        "device": str(device),
        "batch_size": args.batch_size,
        "feature_shape": [args.trajectories, args.steps, args.positions, args.feature_dim],
        "peer_count": num_agents - 1,
        "selector_topk_trajectories": args.selector_topk_trajectories,
        "selector_temporal_pool_steps": args.selector_temporal_pool_steps,
        "attention_heads": args.attention_heads,
        "synthetic_input": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=[*sorted(WORKLOADS), "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--positions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--selector-alpha", type=float, default=0.2)
    parser.add_argument("--selector-topk-trajectories", type=int, default=None)
    parser.add_argument("--selector-temporal-pool-steps", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--gate-type", choices=("single-layer", "two-layers"), default="two-layers")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", type=Path, default=Path("tmp/overhead_module_breakdown.json"))
    args = parser.parse_args(argv)
    if args.feature_dim % args.attention_heads != 0:
        parser.error("--feature-dim must be divisible by --attention-heads")
    if not 0.0 < args.selector_alpha <= 1.0:
        parser.error("--selector-alpha must be in (0, 1]")
    workload_ids = sorted(WORKLOADS) if args.workload == "all" else [args.workload]
    results = [benchmark_one(args, workload_id) for workload_id in workload_ids]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for result in results:
        print(f"{result['workload']}: module1={result['module_1_sample_ms']:.3f} ms, module2={result['module_2_sample_ms']:.3f} ms")
    print(args.output_json)


if __name__ == "__main__":
    main()
