#!/usr/bin/env python3
"""Benchmark ActExchanger's local feature transfer and feature aggregator."""

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

from ours.de_feature_fusion.feature_aggregator import FeatureAggregatorModule


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


def benchmark_one(args: argparse.Namespace, workload_id: str) -> dict:
    workload, num_agents = WORKLOADS[workload_id]
    device = torch.device(args.device)
    source_device = torch.device(args.source_device)
    generator = torch.Generator(device=source_device).manual_seed(args.seed)
    bank = torch.randn(
        args.trajectories * args.steps * args.positions,
        args.feature_dim,
        device=source_device,
        generator=generator,
    )
    indices = torch.arange(args.trajectories * args.steps * args.positions, device=source_device)

    def sample_and_transfer():
        selected = torch.index_select(bank, 0, indices[: args.sample_count])
        return selected.clone().to(device=device, non_blocking=False)

    module_1_times = measure(sample_and_transfer, device, args.warmup, args.samples)
    local = torch.randn(args.batch_size, args.feature_dim, device=device, generator=torch.Generator(device=device).manual_seed(args.seed + 1))
    remote = torch.randn(args.trajectories, args.steps, args.feature_dim, device=device, generator=torch.Generator(device=device).manual_seed(args.seed + 2))
    aggregator = FeatureAggregatorModule(
        local_feature_dim=args.feature_dim,
        remote_feature_dim=args.feature_dim,
        attention_num_heads=args.attention_heads,
        gate_type=args.gate_type,
    ).to(device)
    aggregator.eval()

    def aggregate():
        return aggregator.fuse_feature_stream(local, remote)

    with torch.inference_mode():
        module_2_times = measure(aggregate, device, args.warmup, args.samples)
    return {
        "workload": workload,
        "workload_id": workload_id,
        "num_agents": num_agents,
        "module_1_name": "Local feature sample + transfer",
        "module_2_name": "Feature aggregator",
        "module_1_sample_ms": statistics.median(module_1_times),
        "module_1_mean_ms": statistics.mean(module_1_times),
        "module_2_sample_ms": statistics.median(module_2_times),
        "module_2_mean_ms": statistics.mean(module_2_times),
        "module_1_samples_ms": module_1_times,
        "module_2_samples_ms": module_2_times,
        "device": str(device),
        "source_device": str(source_device),
        "batch_size": args.batch_size,
        "feature_shape": [args.trajectories, args.steps, args.positions, args.feature_dim],
        "sample_count": args.sample_count,
        "attention_heads": args.attention_heads,
        "synthetic_input": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=[*sorted(WORKLOADS), "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-device", default="cpu")
    parser.add_argument("--feature-dim", type=int, default=384)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--positions", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--gate-type", choices=("single-layer", "two-layers"), default="two-layers")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", type=Path, default=Path("tmp/overhead_module_breakdown.json"))
    args = parser.parse_args(argv)
    if args.feature_dim % args.attention_heads != 0:
        parser.error("--feature-dim must be divisible by --attention-heads")
    if args.sample_count > args.trajectories * args.steps * args.positions:
        parser.error("--sample-count exceeds the local feature bank")
    workload_ids = sorted(WORKLOADS) if args.workload == "all" else [args.workload]
    results = [benchmark_one(args, workload_id) for workload_id in workload_ids]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    for result in results:
        print(f"{result['workload']}: module1={result['module_1_sample_ms']:.3f} ms, module2={result['module_2_sample_ms']:.3f} ms")
    print(args.output_json)


if __name__ == "__main__":
    main()
