#!/usr/bin/env python3
"""Plot the maximum supported model-size discussion for four MARL workloads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOADS = [
    ("object_picking_placing", "Object picking and placing", ["AGX Xavier 32GB", "AGX Xavier 32GB"]),
    ("object_stacking", "Object stacking", ["AGX Orin 32GB", "AGX Orin 32GB"]),
    ("cucumber_placing", "Cucumber placing", ["AGX Xavier 32GB", "AGX Orin 32GB", "AGX Orin 32GB"]),
    ("cylinder_transfer", "Cylinder transfer", ["AGX Xavier 32GB", "AGX Xavier 32GB", "AGX Orin 32GB", "AGX Orin 32GB", "AGX Orin 64GB"]),
]
PAPER_CAPACITY_GB = {"AGX Xavier 32GB": 11.3, "AGX Orin 32GB": 11.3, "AGX Orin 64GB": 22.4}
PAPER_KNOWLEDGE_MB = {
    "object_picking_placing": 2.0,
    "object_stacking": 2.0,
    "cucumber_placing": 12.0,
    "cylinder_transfer": 43.0,
}
PAPER_MODULE_OVERHEAD_MB = {
    "object_picking_placing": 25.0,
    "object_stacking": 25.0,
    "cucumber_placing": 70.0,
    "cylinder_transfer": 130.0,
}


def load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        rows = []
        for key, label, devices in WORKLOADS:
            capacities = [PAPER_CAPACITY_GB[d] for d in devices]
            rows.append({
                "workload": key,
                "label": label,
                "devices": devices,
                "max_supported_model_gb": min(capacities),
                "largest_platform_model_gb": max(capacities),
                "knowledge_mb": PAPER_KNOWLEDGE_MB[key],
                "module_overhead_mb": PAPER_MODULE_OVERHEAD_MB[key],
                "source": "paper_reported_capacity",
            })
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("rows", []))
    if not isinstance(payload, list) or not payload:
        raise ValueError("input JSON must contain a non-empty results list")
    return [dict(row) for row in payload]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["workload", "label", "max_supported_model_gb", "largest_platform_model_gb", "knowledge_mb", "module_overhead_mb", "source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("tmp/discussion_model_size_limit.png"))
    parser.add_argument("--summary-output", type=Path, default=Path("tmp/discussion_model_size_limit.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("tmp/discussion_model_size_limit.csv"))
    args = parser.parse_args()
    rows = load_rows(args.input_json)
    labels = [row.get("label", row.get("workload", "unknown")) for row in rows]
    bottleneck = np.asarray([float(row["max_supported_model_gb"]) for row in rows])
    largest = np.asarray([float(row.get("largest_platform_model_gb", row["max_supported_model_gb"])) for row in rows])
    knowledge = np.asarray([float(row.get("knowledge_mb", 0.0)) for row in rows])
    modules = np.asarray([float(row.get("module_overhead_mb", 0.0)) for row in rows])
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    width = 0.36
    axes[0].barh(y - width / 2, bottleneck, height=width, color="#2563eb", label="Workload bottleneck")
    axes[0].barh(y + width / 2, largest, height=width, color="#78a6d8", label="Largest assigned device")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Maximum supported model size (GB)")
    axes[0].set_title("Model capacity across agents")
    axes[0].grid(axis="x", alpha=0.28)
    axes[0].legend(frameon=False)
    axes[1].barh(y - width / 2, knowledge, height=width, color="#66a182", label="Action-grained knowledge")
    axes[1].barh(y + width / 2, modules, height=width, color="#d99058", label="Generation + aggregation")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Additional memory (MB)")
    axes[1].set_title("ActExchanger memory overhead")
    axes[1].grid(axis="x", alpha=0.28)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#555555")
    fig.suptitle("Maximum supported model size in multi-agent workloads", y=1.01)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_output, rows)
    print(args.output)
    print(args.summary_output)
    print(args.csv_output)
    for row in rows:
        print(f"{row.get('label', row.get('workload'))}: bottleneck={float(row['max_supported_model_gb']):.1f} GB")


if __name__ == "__main__":
    main()
