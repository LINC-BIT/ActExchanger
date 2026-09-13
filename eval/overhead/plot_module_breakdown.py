#!/usr/bin/env python3
"""Plot the two ActExchanger overhead modules from benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOAD_ORDER = [
    "Object picking and placing",
    "Object stacking",
    "Cucumber placing",
    "Cylinder transfer",
]


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("rows", [payload]))
    return payload if isinstance(payload, list) else []


def _float(row: dict, key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def plot_breakdown(rows: list[dict], output_path: Path, *, colored: bool) -> None:
    by_workload = {str(row.get("workload")): row for row in rows}
    visible = [name for name in WORKLOAD_ORDER if name in by_workload]
    if not visible:
        raise ValueError("No recognized workload rows in benchmark output")
    module_1 = np.array([_float(by_workload[name], "module_1_sample_ms") or 0.0 for name in visible])
    module_2 = np.array([_float(by_workload[name], "module_2_sample_ms") or 0.0 for name in visible])
    y = np.arange(len(visible), dtype=float) * 0.22
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.18, top=0.94)
    module_1_color = "#8b8b8b" if colored else "#989898"
    module_2_color = "#4c93c7" if colored else "#d1d1d1"
    ax.barh(y, module_1, height=0.14, color=module_1_color, edgecolor="black", linewidth=1.0, label="Action-grained knowledge generator")
    ax.barh(y, module_2, left=module_1, height=0.14, color=module_2_color, edgecolor="black", linewidth=1.0, label="Action-grained knowledge aggregator")
    ax.set_yticks(y)
    ax.set_yticklabels(visible)
    ax.invert_yaxis()
    ax.set_xlabel("Time per sample (ms)")
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.9)
    ax.set_axisbelow(True)
    for yi, first, second in zip(y, module_1, module_2):
        total = first + second
        ax.text(total + max(total * 0.02, 0.01), yi, f"{total:.2f}", va="center", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("tmp/overhead_module_breakdown.png"))
    parser.add_argument("--colored-output", type=Path, default=None)
    args = parser.parse_args(argv)
    input_path = args.input_json or args.input_csv
    if input_path is None:
        parser.error("one of --input-json or --input-csv is required")
    rows = _read_rows(input_path)
    plot_breakdown(rows, args.output, colored=False)
    if args.colored_output is not None:
        plot_breakdown(rows, args.colored_output, colored=True)
    print(args.output)
    if args.colored_output is not None:
        print(args.colored_output)


if __name__ == "__main__":
    main()
