#!/usr/bin/env python3
"""Plot module timing and communication estimates from the discussion simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOAD_LABELS = {
    "object_picking_placing": "Object picking and placing",
    "object_stacking": "Object stacking",
    "cucumber_placing": "Cucumber placing",
    "cylinder_transfer": "Cylinder transfer",
}
WORKLOAD_ORDER = list(WORKLOAD_LABELS)


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    return payload if isinstance(payload, list) else []


def number(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp/discussion_module_simulator.png"))
    args = parser.parse_args()
    rows = {str(row.get("task")): row for row in load_rows(args.input_json)}
    names = [name for name in WORKLOAD_ORDER if name in rows]
    if not names:
        raise ValueError("Input contains no recognized multi-agent workload")

    labels = [WORKLOAD_LABELS[name] for name in names]
    module_1 = np.array([number(rows[name], "module_1_ms") for name in names])
    module_2 = np.array([number(rows[name], "module_2_ms") for name in names])
    communication = np.array([number(rows[name], "estimated_effective_mib_1h") for name in names])
    ratio = np.array([number(rows[name], "effective_ratio") for name in names])
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(y, module_1, color="#989898", edgecolor="black", label="Action-grained feature generator")
    axes[0].barh(y, module_2, left=module_1, color="#4c93c7", edgecolor="black", label="Feature aggregator")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Time per simulated step (ms)")
    axes[0].set_title("Module timing")
    axes[0].grid(axis="x", color="#dddddd")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].barh(y, communication, color="#66a182", edgecolor="black")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Estimated effective communication (MiB/hour)")
    axes[1].set_title("Communication estimate")
    axes[1].grid(axis="x", color="#dddddd")
    for yi, value, value_ratio in zip(y, communication, ratio):
        axes[1].text(value + max(float(communication.max()) * 0.015, 0.01), yi, f"ratio={value_ratio:.3f}", va="center", fontsize=9)
    for ax in axes:
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#666666")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
