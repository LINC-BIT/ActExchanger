#!/usr/bin/env python3
"""Plot VLA and CNN multi-agent accuracy from completed training runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("history", "metrics", "records", "rows", "data"):
            found = records(payload.get(key))
            if found:
                return found
        return [payload]
    return []


def series(run_dir: Path) -> tuple[list[float], list[float]]:
    for name in ("metrics.json", "metrics_history.json", "history.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        xs, ys = [], []
        for index, row in enumerate(records(json.loads(path.read_text(encoding="utf-8")))):
            value = next((row.get(key) for key in ("score", "success_once", "accuracy", "eval_success_once") if row.get(key) is not None), None)
            if value is None:
                continue
            elapsed = row.get("elapsed_minutes", row.get("elapsed_time_minutes", index))
            xs.append(float(elapsed))
            ys.append(float(value) / 100.0 if float(value) > 1 else float(value))
        if xs:
            return xs, ys
    raise FileNotFoundError(f"no accuracy history under {run_dir}")


def draw(vla, cnn, limit: float, output: Path) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(8.1, 4.4))
    result = {}
    for label, color, values in (("VLA", "#F04B4B", vla), ("CNN", "#638DEE", cnn)):
        xs, ys = values
        selected = [(x, y) for x, y in zip(xs, ys) if x <= limit]
        if not selected:
            raise ValueError(f"{label} has no samples before {limit} minutes")
        plot_x, plot_y = zip(*selected)
        ax.plot(plot_x, plot_y, color=color, linewidth=2.2, label=label)
        result[f"{label.lower()}_mean_accuracy"] = sum(plot_y) / len(plot_y)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-run-dir", type=Path, required=True)
    parser.add_argument("--cnn-run-dir", type=Path, required=True)
    parser.add_argument("--output-3h", type=Path, required=True)
    parser.add_argument("--output-10h", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    vla, cnn = series(args.vla_run_dir), series(args.cnn_run_dir)
    summary = {
        "source": "training_metrics",
        "vla_run_dir": str(args.vla_run_dir),
        "cnn_run_dir": str(args.cnn_run_dir),
        "3h": draw(vla, cnn, 180, args.output_3h),
        "10h": draw(vla, cnn, 600, args.output_10h),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.output_3h)
    print(args.output_10h)
    print(args.summary_output)


if __name__ == "__main__":
    main()
