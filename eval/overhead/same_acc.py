#!/usr/bin/env python3
"""Measure wall-clock time required to reach the same accuracy as ours."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKLOADS = {
    "object_picking_placing": "Object picking and placing",
    "object_stacking": "Object stacking",
    "cucumber_placing": "Cucumber placing",
    "cylinder_transfer": "Cylinder transfer",
}
METHOD_FLAGS = {
    "baseline": ("MAPPO", "baseline_run_dir"),
    "dicg": ("DICG", "dicg_run_dir"),
    "mat": ("MAT", "mat_run_dir"),
    "tgcnet": ("TGCNet", "tgcnet_run_dir"),
    "maple": ("MAPLE", "maple_run_dir"),
    "comatrack": ("CoMaTrack", "comatrack_run_dir"),
    "maporl": ("MAPoRL", "maporl_run_dir"),
    "mpdf": ("MPDF", "mpdf_run_dir"),
    "magrpo": ("MAGRPO", "magrpo_run_dir"),
}
DEFAULT_TAG = "eval/success_once"


@dataclass
class Point:
    x: float
    y: float


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("history", "metrics", "records", "data", "rows", "metrics_history"):
        rows = _json_rows(payload.get(key))
        if rows:
            return rows
    if any(key in payload for key in ("score", "accuracy", "success_once", "value")):
        return [payload]
    return []


def _row_value(row: dict[str, Any]) -> float | None:
    for key in ("accuracy", "success_once", "score", "value", "eval_success_once", "eval/success_once"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _row_x(row: dict[str, Any], x_axis: str, index: int) -> float:
    if x_axis == "step":
        for key in ("step", "global_step", "iteration", "update"):
            value = _number(row.get(key))
            if value is not None:
                return value
        return float(index)
    for key in ("minutes", "time_minutes", "elapsed_minutes"):
        value = _number(row.get(key))
        if value is not None:
            return value
    for key in ("seconds", "time_seconds", "elapsed_seconds", "wall_time_seconds"):
        value = _number(row.get(key))
        if value is not None:
            return value / 60.0
    value = _number(row.get("wall_time"))
    if value is not None:
        return value / 60.0
    return float(index)


def _load_tensorboard(run_dir: Path, tag: str, x_axis: str) -> list[Point]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return []
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return []
    try:
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir), size_guidance={event_accumulator.SCALARS: 0}
        )
        accumulator.Reload()
        events = accumulator.Scalars(tag)
    except (OSError, KeyError, ValueError):
        return []
    if not events:
        return []
    base_wall_time = events[0].wall_time
    points = []
    for event in events:
        x = event.step if x_axis == "step" else (event.wall_time - base_wall_time) / 60.0
        points.append(Point(float(x), float(event.value)))
    return points


def _load_json_curve(run_dir: Path, x_axis: str) -> list[Point]:
    for filename in ("metrics_history.json", "metrics.json", "history.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        rows = _json_rows(_read_json(path))
        grouped: dict[float, list[float]] = {}
        for index, row in enumerate(rows):
            value = _row_value(row)
            if value is not None:
                grouped.setdefault(_row_x(row, x_axis, index), []).append(value)
        points = [Point(x, sum(values) / len(values)) for x, values in grouped.items()]
        if points:
            if x_axis == "time" and points:
                base_x = min(point.x for point in points)
                points = [Point(point.x - base_x, point.y) for point in points]
            return sorted(points, key=lambda point: point.x)
    return []


def load_curve(run_dir: Path | None, tag: str, x_axis: str) -> list[Point]:
    if run_dir is None:
        return []
    return _load_tensorboard(run_dir, tag, x_axis) or _load_json_curve(run_dir, x_axis)


def has_wall_clock_axis(run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    if (run_dir / "tb").exists():
        return True
    for filename in ("metrics_history.json", "metrics.json", "history.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        rows = _json_rows(_read_json(path))
        time_keys = {"minutes", "time_minutes", "elapsed_minutes", "seconds", "time_seconds", "elapsed_seconds", "wall_time_seconds", "wall_time"}
        return any(time_keys.intersection(row) for row in rows)
    return False


def target_accuracy(points: list[Point], mode: str, topk: int) -> float | None:
    values = [point.y for point in points if math.isfinite(point.y)]
    if not values:
        return None
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    values.sort(reverse=True)
    return sum(values[: max(1, min(topk, len(values)))]) / max(1, min(topk, len(values)))


def first_reach(points: list[Point], target: float | None, tolerance: float) -> float | None:
    if target is None:
        return None
    for point in sorted(points, key=lambda item: item.x):
        if point.y + tolerance >= target:
            return point.x
    return None


def _relative(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or value <= 0:
        return None
    return reference / value


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], float | None]:
    ours = load_curve(args.run_dir, args.tag, args.x_axis)
    target = target_accuracy(ours, args.target_mode, args.topk)
    methods: list[tuple[str, Path | None]] = [("Ours", args.run_dir)]
    for _, (label, attribute) in METHOD_FLAGS.items():
        methods.append((label, getattr(args, attribute)))
    ours_time = first_reach(ours, target, args.tolerance)
    time_unit = "steps" if args.x_axis == "step" or not has_wall_clock_axis(args.run_dir) else "minutes"
    rows = []
    for label, run_dir in methods:
        points = ours if label == "Ours" else load_curve(run_dir, args.tag, args.x_axis)
        reached = ours_time if label == "Ours" else first_reach(points, target, args.tolerance)
        rows.append(
            {
                "workload": WORKLOADS[args.workload],
                "workload_id": args.workload,
                "method": label,
                "run_dir": "" if run_dir is None else str(run_dir),
                "target_accuracy": target,
                "reach_time": reached,
                "time_unit": time_unit,
                "speedup_vs_ours": _relative(reached, ours_time),
                "status": "missing_curve" if not points else ("reached" if reached is not None else "not_reached"),
            }
        )
    return rows, target


def write_outputs(rows: list[dict[str, Any]], csv_path: Path, json_path: Path, args: argparse.Namespace) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["workload", "method", "target_accuracy", "reach_time"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "workload": WORKLOADS[args.workload],
        "workload_id": args.workload,
        "tag": args.tag,
        "x_axis": args.x_axis,
        "target_mode": args.target_mode,
        "target_accuracy": rows[0].get("target_accuracy") if rows else None,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--run-dir", type=Path, required=True, help="ActExchanger run directory")
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    for flag, (label, _) in METHOD_FLAGS.items():
        if flag == "baseline":
            continue
        parser.add_argument(f"--{flag}-run-dir", type=Path, default=None, metavar="DIR", help=f"{label} run directory")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--x-axis", choices=("time", "step"), default="time")
    parser.add_argument("--target-mode", choices=("max", "mean", "topk_mean"), default="max")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    default_stem = args.workload
    csv_path = args.output_csv or Path("tmp") / f"overhead_same_acc_{default_stem}.csv"
    json_path = args.output_json or Path("tmp") / f"overhead_same_acc_{default_stem}.json"
    rows, target = build_rows(args)
    write_outputs(rows, csv_path, json_path, args)
    print(f"target_accuracy={target}")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
