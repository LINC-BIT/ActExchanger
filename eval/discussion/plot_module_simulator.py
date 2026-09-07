#!/usr/bin/env python3
import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AgentTimelineSpec:
    name: str
    stages: Sequence[Tuple[str, float]]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    max_episode_steps: int
    hidden_dim: int
    attention_heads: int
    dtype_bytes: int
    generator_tokens: int
    aggregator_tokens: int
    agent_specs: Sequence[AgentTimelineSpec]


TASK_SPECS: Dict[str, TaskSpec] = {
    "object_picking_placing": TaskSpec(
        name="object_picking_placing",
        max_episode_steps=100,
        hidden_dim=384,
        attention_heads=6,
        dtype_bytes=4,
        generator_tokens=8,
        aggregator_tokens=12,
        agent_specs=(
            AgentTimelineSpec("left", (("idle", 0.08), ("approach", 0.28), ("push", 0.42), ("return", 0.22))),
            AgentTimelineSpec("right", (("idle", 0.08), ("approach", 0.24), ("grasp", 0.16), ("move", 0.30), ("return", 0.22))),
        ),
    ),
    "object_stacking": TaskSpec(
        name="object_stacking",
        max_episode_steps=100,
        hidden_dim=384,
        attention_heads=6,
        dtype_bytes=4,
        generator_tokens=8,
        aggregator_tokens=12,
        agent_specs=(
            AgentTimelineSpec("left", (("idle", 0.08), ("approach", 0.22), ("grasp", 0.18), ("stack", 0.30), ("return", 0.22))),
            AgentTimelineSpec("right", (("idle", 0.08), ("approach", 0.20), ("grasp", 0.18), ("stack", 0.32), ("return", 0.22))),
        ),
    ),
    "cucumber_placing": TaskSpec(
        name="cucumber_placing",
        max_episode_steps=100,
        hidden_dim=384,
        attention_heads=6,
        dtype_bytes=4,
        generator_tokens=10,
        aggregator_tokens=16,
        agent_specs=(
            AgentTimelineSpec("center", (("idle", 0.10), ("approach", 0.25), ("open_lid", 0.20), ("place", 0.25), ("return", 0.20))),
            AgentTimelineSpec("left", (("idle", 0.10), ("approach", 0.20), ("grasp", 0.18), ("move", 0.22), ("place", 0.18), ("return", 0.12))),
            AgentTimelineSpec("right", (("idle", 0.12), ("approach", 0.22), ("grasp", 0.16), ("move", 0.20), ("place", 0.18), ("return", 0.12))),
        ),
    ),
    "cylinder_transfer": TaskSpec(
        name="cylinder_transfer",
        max_episode_steps=300,
        hidden_dim=320,
        attention_heads=8,
        dtype_bytes=4,
        generator_tokens=12,
        aggregator_tokens=20,
        agent_specs=(
            AgentTimelineSpec("panda", (("idle", 0.08), ("approach", 0.16), ("grasp", 0.18), ("pass", 0.34), ("return", 0.24))),
            AgentTimelineSpec("so100", (("idle", 0.12), ("approach", 0.15), ("push", 0.20), ("pass", 0.28), ("return", 0.25))),
            AgentTimelineSpec("widowx", (("idle", 0.16), ("approach", 0.15), ("receive", 0.18), ("pass", 0.22), ("return", 0.29))),
            AgentTimelineSpec("xarm6", (("idle", 0.20), ("approach", 0.12), ("receive", 0.16), ("pass", 0.20), ("return", 0.32))),
            AgentTimelineSpec("fixed_inspire_hand", (("idle", 0.28), ("grasp", 0.18), ("hold", 0.22), ("return", 0.32))),
        ),
    ),
}


class ActionGrainedFeatureGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAggregator(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, local_feature: torch.Tensor, remote_features: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(local_feature).unsqueeze(1)
        attn_out, _ = self.attn(query=query, key=remote_features, value=remote_features, need_weights=False)
        attn_out = self.out_norm(attn_out.squeeze(1))
        gate = torch.sigmoid(self.gate(torch.cat([local_feature, attn_out], dim=-1)))
        return local_feature + gate * attn_out


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-4, None)
    return weights / weights.sum()


def _expand_stages(total_steps: int, stage_spec: AgentTimelineSpec, rng: np.random.Generator) -> List[str]:
    weights = np.array([w for _, w in stage_spec.stages], dtype=np.float64)
    weights = _normalize_weights(weights * (1.0 + rng.normal(0.0, 0.06, size=weights.shape)))
    durations = np.maximum(1, np.floor(weights * total_steps).astype(int))
    diff = int(total_steps - durations.sum())
    durations[-1] += diff
    if durations[-1] <= 0:
        deficit = 1 - durations[-1]
        durations[-1] = 1
        for idx in range(len(durations) - 2, -1, -1):
            take = min(deficit, max(0, durations[idx] - 1))
            durations[idx] -= take
            deficit -= take
            if deficit == 0:
                break
    timeline: List[str] = []
    for (stage_name, _), duration in zip(stage_spec.stages, durations.tolist()):
        timeline.extend([stage_name] * int(duration))
    if len(timeline) < total_steps:
        timeline.extend([stage_spec.stages[-1][0]] * (total_steps - len(timeline)))
    return timeline[:total_steps]


def simulate_timelines(task: TaskSpec, episodes: int, seed: int) -> List[List[List[str]]]:
    rng = np.random.default_rng(seed)
    return [
        [_expand_stages(task.max_episode_steps, agent_spec, rng) for agent_spec in task.agent_specs]
        for _ in range(episodes)
    ]


def compute_effective_bytes(task: TaskSpec, timelines: List[List[List[str]]]) -> Dict[str, float]:
    feature_bytes = task.hidden_dim * task.dtype_bytes
    total_direct_bytes = 0
    effective_bytes = 0
    total_pair_units = 0
    effective_pair_units = 0

    for episode in timelines:
        for step in range(task.max_episode_steps):
            labels = [agent_timeline[step] for agent_timeline in episode]
            pair_count = 0
            for i, j in permutations(range(len(labels)), 2):
                total_pair_units += 1
                total_direct_bytes += feature_bytes
                if labels[i] == labels[j]:
                    pair_count += 1
                    effective_pair_units += 1
                    effective_bytes += feature_bytes
            # keep pair_count observable for debugging if needed
            _ = pair_count

    ratio = float(effective_bytes) / float(total_direct_bytes) if total_direct_bytes else 0.0
    return {
        "feature_bytes_per_packet": float(feature_bytes),
        "total_direct_bytes": float(total_direct_bytes),
        "effective_bytes": float(effective_bytes),
        "total_pair_units": float(total_pair_units),
        "effective_pair_units": float(effective_pair_units),
        "effective_ratio": float(ratio),
    }


@torch.inference_mode()
def benchmark_modules(task: TaskSpec, device: torch.device, batch_size: int, warmup: int, iters: int) -> Dict[str, float]:
    generator = ActionGrainedFeatureGenerator(task.hidden_dim, task.hidden_dim).to(device=device, dtype=torch.float32)
    aggregator = FeatureAggregator(task.hidden_dim, task.attention_heads).to(device=device, dtype=torch.float32)

    gen_in = torch.randn(batch_size, task.generator_tokens, task.hidden_dim, device=device, dtype=torch.float32)
    local_feature = torch.randn(batch_size, task.hidden_dim, device=device, dtype=torch.float32)
    remote_features = torch.randn(
        batch_size,
        max(1, task.aggregator_tokens * max(len(task.agent_specs) - 1, 1)),
        task.hidden_dim,
        device=device,
        dtype=torch.float32,
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        _ = generator(gen_in)
        _ = aggregator(local_feature, remote_features)
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = generator(gen_in)
    sync()
    gen_ms = (time.perf_counter() - t0) * 1000.0 / max(iters, 1)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = aggregator(local_feature, remote_features)
    sync()
    agg_ms = (time.perf_counter() - t0) * 1000.0 / max(iters, 1)

    iteration_ms = float(gen_ms + agg_ms)
    return {
        "module_1_name": "action-grained feature generator",
        "module_2_name": "feature aggregator",
        "module_1_ms": float(gen_ms),
        "module_2_ms": float(agg_ms),
        "iteration_ms": iteration_ms,
        "module_1_pct": float(gen_ms / iteration_ms) if iteration_ms > 0 else 0.0,
        "module_2_pct": float(agg_ms / iteration_ms) if iteration_ms > 0 else 0.0,
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(task_name: str, args: argparse.Namespace) -> Dict[str, object]:
    task = TASK_SPECS[task_name]
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    timelines = simulate_timelines(task, args.episodes, args.seed)
    effective = compute_effective_bytes(task, timelines)
    timings = benchmark_modules(task, device, args.batch_size, args.warmup, args.iters)

    packets_per_step = len(task.agent_specs) * max(len(task.agent_specs) - 1, 0)
    total_steps = task.max_episode_steps * args.episodes
    steps_per_sec = 1000.0 / max(timings["iteration_ms"], 1e-6)
    total_direct_per_step = effective["feature_bytes_per_packet"] * packets_per_step
    effective_per_step = effective["feature_bytes_per_packet"] * effective["effective_ratio"] * packets_per_step

    return {
        "task": task.name,
        "device": str(device),
        "episodes": args.episodes,
        "steps": total_steps,
        "module_1_name": timings["module_1_name"],
        "module_2_name": timings["module_2_name"],
        "module_1_ms": timings["module_1_ms"],
        "module_2_ms": timings["module_2_ms"],
        "iteration_ms": timings["iteration_ms"],
        "module_1_pct": timings["module_1_pct"],
        "module_2_pct": timings["module_2_pct"],
        "estimated_steps_per_sec": steps_per_sec,
        "effective_bytes": effective["effective_bytes"],
        "effective_ratio": effective["effective_ratio"],
        "estimated_effective_mib_1h": effective_per_step * steps_per_sec * 3600.0 / (1024.0 * 1024.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_SPECS.keys()) + ["all"], default="all")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = list(TASK_SPECS.keys()) if args.task == "all" else [args.task]
    results = [run(task_name, args) for task_name in tasks]
    payload = results if len(results) > 1 else results[0]
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
