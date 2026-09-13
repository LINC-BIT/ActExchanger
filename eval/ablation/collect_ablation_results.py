#!/usr/bin/env python3
"""Collect ablation accuracies from completed experiment runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GROUPS = {
    "Feature length": ("Constraining action-grained feature length by", ["Original length", "MDL"]),
    "Feature generation": (
        "Generating action-grained features using",
        ["Raw features", "Semantic-only features", "Spatial-only features", "Semantic-spatial hybrid features"],
    ),
    "Knowledge representation": (
        "Representing action-grained features as",
        ["Multiple unfused features per action", "Single fused feature per action"],
    ),
    "Aggregation strategy": (
        "Aggregation strategy",
        ["No aggregation", "Current-forward aggregation", "Random aggregation", "Same-action aggregation"],
    ),
}


def number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("history", "metrics", "records", "data", "rows", "metrics_history"):
        rows = rows_from_payload(payload.get(key))
        if rows:
            return rows
    return [payload]


def score_from_run(run_dir: Path) -> float:
    for filename in ("eval_metrics.json", "metrics.json", "metrics_history.json", "history.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read {path}") from error
        values = []
        for row in rows_from_payload(payload):
            for key in ("score", "success_once", "accuracy", "eval_success_once", "eval/success_once"):
                value = number(row.get(key))
                if value is not None:
                    values.append(value)
                    break
        if values:
            return max(values)
    raise FileNotFoundError(f"No accuracy history found under {run_dir}")


def parse_run(value: str) -> tuple[str, str, Path]:
    try:
        group, choice, path = value.split("::", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected GROUP::CHOICE::RUN_DIR") from error
    if group not in GROUPS or choice not in GROUPS[group][1]:
        raise argparse.ArgumentTypeError(f"unknown ablation entry: {group} / {choice}")
    return group, choice, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    paths = {(group, choice): path for group, choice, path in args.run}
    rows = []
    for group, (type_name, choices) in GROUPS.items():
        for choice in choices:
            path = paths.get((group, choice))
            if path is None:
                parser.error(f"missing run for {group} / {choice}")
            rows.append(
                {
                    "type": type_name,
                    "choice": choice,
                    "accuracy": score_from_run(path),
                    "run_dir": str(path),
                }
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
