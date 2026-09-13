#!/usr/bin/env python3
"""Collect multi-agent ICL histories from two completed run directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("history", "metrics", "records", "rows", "data"):
            found = rows(payload.get(key))
            if found:
                return found
        return [payload]
    return []


def load_history(run_dir: Path) -> list[float]:
    for name in ("metrics.json", "metrics_history.json", "history.json", "eval_metrics.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = []
        for item in rows(payload):
            for key in ("score", "success_once", "accuracy", "eval_success_once", "eval/success_once"):
                value = item.get(key)
                if value is not None:
                    value = float(value)
                    if value > 1:
                        value /= 100
                    if 0 <= value <= 1:
                        values.append(value)
                    break
        if values:
            return values
    raise FileNotFoundError(f"no accuracy history found under {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True, help="WORKLOAD::OURS_DIR::BASELINE_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for entry in args.entry:
        try:
            workload, ours, baseline = entry.split("::", 2)
        except ValueError as error:
            raise SystemExit(f"invalid --entry: {entry}") from error
        result[workload] = {"actexchanger": load_history(Path(ours)), "ricl": load_history(Path(baseline))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
