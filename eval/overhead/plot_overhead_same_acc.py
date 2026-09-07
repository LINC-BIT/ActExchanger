#!/usr/bin/env python3
"""Plot time-to-same-accuracy results produced by same_acc.py."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("results", []))
    return payload if isinstance(payload, list) else []


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def plot(rows: list[dict], output: Path, title: str | None) -> None:
    rows = [row for row in rows if number(row.get("reach_time")) is not None]
    if not rows:
        raise ValueError("No reached methods found in input")
    labels = [str(row.get("method", "unknown")) for row in rows]
    values = np.array([number(row["reach_time"]) for row in rows], dtype=float)
    unit = str(rows[0].get("time_unit", "minutes"))
    colors = ["#F04B4B" if label == "Ours" else "#7B8794" for label in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Time to reach same accuracy ({unit})")
    ax.set_ylabel("Method")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for yi, value in zip(y, values):
        ax.text(value + max(float(values.max()) * 0.015, 0.01), yi, f"{value:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("tmp/overhead_same_acc.png"))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    input_path = args.input_json or args.input_csv
    if input_path is None:
        parser.error("one of --input-json or --input-csv is required")
    plot(load_rows(input_path), args.output, args.title)
    print(args.output)


if __name__ == "__main__":
    main()
