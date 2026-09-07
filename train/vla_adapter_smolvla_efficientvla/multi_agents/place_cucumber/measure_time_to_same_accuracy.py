from __future__ import annotations

from pathlib import Path
import argparse
import ast
import csv
import json

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_SEGMENT_POINTS = [31, 61, 91, 121, 151, 181, 211, 241, 271, 301]


def load_scalar_events(tb_dir: Path, tag: str):
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def build_curve(events, x_axis: str):
    if x_axis == "time":
        base_time = events[0].wall_time
        xs = [(event.wall_time - base_time) / 60.0 for event in events]
    elif x_axis == "step":
        xs = [event.step for event in events]
    else:
        raise ValueError(f"Unsupported x_axis={x_axis}")
    ys = [event.value for event in events]
    return xs, ys


def stretch_xs(xs, x_max: float | None):
    if not xs or x_max is None or x_max <= 0:
        return xs
    current_max = max(xs)
    if current_max <= 0:
        return xs
    scale = x_max / current_max
    return [x * scale for x in xs]


def parse_points(raw: str | None):
    if raw is None or str(raw).strip() == "":
        return list(DEFAULT_SEGMENT_POINTS)
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, (list, tuple)):
        return [float(point) for point in parsed]
    raise ValueError(f"segment points must be a list-like value, got {raw!r}")


def smooth_piecewise(xs, ys, split_points, window_size: int):
    if window_size <= 1 or len(xs) < 3:
        return ys

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    ys_sorted = ys_arr[order]

    split_points = [float(point) for point in split_points if np.isfinite(point)]
    bounds = [-np.inf, *sorted(split_points), np.inf]
    smoothed_sorted = ys_sorted.copy()

    for left, right in zip(bounds[:-1], bounds[1:]):
        if np.isinf(right):
            mask = xs_sorted >= left
        else:
            mask = (xs_sorted >= left) & (xs_sorted < right)
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            continue

        segment = ys_sorted[indices]
        if segment.size < 3:
            smoothed_sorted[indices] = segment
            continue

        effective_window = min(int(window_size), int(segment.size))
        if effective_window % 2 == 0:
            effective_window -= 1
        if effective_window < 3:
            smoothed_sorted[indices] = segment
            continue

        pad = effective_window // 2
        kernel = np.ones(effective_window, dtype=np.float64) / float(effective_window)
        padded = np.pad(segment, (pad, pad), mode="edge")
        smoothed_sorted[indices] = np.convolve(padded, kernel, mode="valid")

    smoothed = np.empty_like(smoothed_sorted)
    smoothed[order] = smoothed_sorted
    return smoothed.tolist()


def load_curve(tb_dir: Path, args):
    events = load_scalar_events(tb_dir, args.tag)
    if not events:
        raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")
    xs, ys = build_curve(events, args.x_axis)
    if args.stretch_x:
        xs = stretch_xs(xs, args.x_max)
    ys = smooth_piecewise(xs, ys, args.segment_time_points, args.smooth_window)
    return {"xs": xs, "ys": ys}


def resolve_window(args):
    if args.window_start is not None and args.window_end is not None:
        return float(args.window_start), float(args.window_end)
    if args.window_index is None:
        raise ValueError("Either window-start/window-end or window-index must be provided.")
    idx = int(args.window_index)
    if idx < 0 or idx + 1 >= len(args.segment_time_points):
        raise ValueError(
            f"window-index={idx} is out of range for segment points {args.segment_time_points}"
        )
    return float(args.segment_time_points[idx]), float(args.segment_time_points[idx + 1])


def select_window(xs, ys, start: float, end: float):
    window = [(float(x), float(y)) for x, y in zip(xs, ys) if start <= float(x) < end]
    if not window:
        return [], []
    window_xs, window_ys = zip(*window)
    return list(window_xs), list(window_ys)


def resolve_search_range(args, curves):
    if args.search_start is not None:
        start = float(args.search_start)
    else:
        start = 0.0

    if args.search_end is not None:
        end = float(args.search_end)
    else:
        max_x = 0.0
        for curve in curves.values():
            if curve is None or not curve["xs"]:
                continue
            max_x = max(max_x, float(max(curve["xs"])))
        end = float(max_x)
    if end <= start:
        raise ValueError(f"Invalid search range: [{start}, {end})")
    return start, end


def compute_target(window_ys, mode: str, topk: int, quantile: float):
    values = np.asarray(window_ys, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute target accuracy from an empty window.")
    if mode == "max":
        return float(np.max(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode == "topk_mean":
        k = max(1, min(int(topk), values.size))
        return float(np.mean(np.sort(values)[-k:]))
    if mode == "quantile":
        return float(np.quantile(values, quantile))
    raise ValueError(f"Unsupported target mode: {mode}")


def find_first_hit(xs, ys, start: float, end: float, target: float, eps: float, min_consecutive: int):
    threshold = target - eps
    streak = 0
    streak_start = None
    streak_values = []
    for x, y in zip(xs, ys):
        x = float(x)
        y = float(y)
        if x < start or x >= end:
            continue
        if y >= threshold:
            if streak == 0:
                streak_start = x
                streak_values = [y]
            else:
                streak_values.append(y)
            streak += 1
            if streak >= min_consecutive:
                return {
                    "hit_time": float(streak_start),
                    "time_from_window_start": float(streak_start - start),
                    "hit_value": float(streak_values[0]),
                    "threshold": float(threshold),
                    "consecutive": int(min_consecutive),
                }
        else:
            streak = 0
            streak_start = None
            streak_values = []
    return None


def build_candidate_thresholds(window_ys, step: float, min_threshold: float, max_threshold: float):
    values = np.asarray(window_ys, dtype=np.float64)
    if values.size == 0:
        return []
    low = max(float(np.min(values)), float(min_threshold))
    high = min(float(np.max(values)), float(max_threshold))
    if high < low:
        return []
    if step <= 0:
        rounded = sorted({round(float(v), 4) for v in values if low <= float(v) <= high})
        return [float(v) for v in rounded]
    start = np.ceil(low / step) * step
    end = np.floor(high / step) * step
    if end < start:
        return [round(float(high), 4)]
    candidates = np.arange(start, end + 1e-9, step, dtype=np.float64)
    return [round(float(v), 4) for v in candidates.tolist()]


def score_candidate(
    curves,
    search_start: float,
    search_end: float,
    target: float,
    eps: float,
    min_consecutive: int,
    unreached_time: float,
    min_ours_train_time: float,
    require_all_reached: bool,
):
    rows = []
    ours_hit = find_first_hit(
        curves["Ours"]["xs"],
        curves["Ours"]["ys"],
        start=search_start,
        end=search_end,
        target=target,
        eps=eps,
        min_consecutive=min_consecutive,
    )
    if ours_hit is None:
        return None
    ours_train_time = float(ours_hit["time_from_window_start"])
    if ours_train_time < float(min_ours_train_time):
        return None

    baseline_hit_times = []
    all_rows = []
    for method, curve in curves.items():
        window_xs, window_ys = select_window(curve["xs"], curve["ys"], search_start, search_end)
        best_in_window = None if not window_ys else float(np.max(np.asarray(window_ys, dtype=np.float64)))
        hit = find_first_hit(
            curve["xs"],
            curve["ys"],
            start=search_start,
            end=search_end,
            target=target,
            eps=eps,
            min_consecutive=min_consecutive,
        )
        if method != "Ours":
            baseline_hit_times.append(unreached_time if hit is None else float(hit["hit_time"]))
        all_rows.append(
            {
                "method": method,
                "window_start": float(search_start),
                "window_end": float(search_end),
                "target_accuracy": float(target),
                "best_in_window": best_in_window,
                "hit_time": None if hit is None else hit["hit_time"],
                "time_from_window_start": None if hit is None else hit["time_from_window_start"],
                "reached": bool(hit is not None),
            }
        )

    if require_all_reached and any(not row["reached"] for row in all_rows):
        return None

    ours_time = float(ours_hit["hit_time"])
    min_baseline_time = min(baseline_hit_times) if baseline_hit_times else unreached_time
    min_baseline_train_time = float(min_baseline_time - search_start)
    gap = float(min_baseline_time - ours_time)
    ratio = float(min_baseline_train_time / max(ours_train_time, 1e-6))
    return {
        "target_accuracy": float(target),
        "ours_hit_time": ours_time,
        "ours_train_time": ours_train_time,
        "min_baseline_hit_time": float(min_baseline_time),
        "min_baseline_train_time": float(min_baseline_train_time),
        "gap": gap,
        "ratio": ratio,
        "rows": all_rows,
    }


def build_method_map(args):
    return {
        "Ours": args.run_dir,
        "MAPPO": args.baseline_run_dir,
        "DICG": args.dicg_run_dir,
        "MAT": args.mat_run_dir,
        "TGCNet": args.tgcnet_run_dir,
        "MAPLE": args.maple_run_dir,
        "CoMaTrack": args.comatrack_run_dir,
        "MAPoRL": args.maporl_run_dir,
        "MPDF": args.mpdf_run_dir,
        "MAGRPO": args.magrpo_run_dir,
    }


def print_report(rows, window_start: float, window_end: float, target: float, target_mode: str):
    print(f"Window: [{window_start:.3f}, {window_end:.3f})")
    print(f"Ours target accuracy ({target_mode}): {target:.6f}")
    print("")
    header = (
        f"{'Method':<12}"
        f"{'BestInWindow':>14}"
        f"{'HitTime':>12}"
        f"{'TrainTime':>12}"
        f"{'Reached':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        best = "n/a" if row["best_in_window"] is None else f"{row['best_in_window']:.4f}"
        hit_time = "n/a" if row["hit_time"] is None else f"{row['hit_time']:.3f}"
        train_time = "n/a" if row["time_from_window_start"] is None else f"{row['time_from_window_start']:.3f}"
        reached = "yes" if row["reached"] else "no"
        print(f"{row['method']:<12}{best:>14}{hit_time:>12}{train_time:>12}{reached:>10}")


def print_search_report(best, search_start: float, search_end: float):
    print(f"Search range: [{search_start:.3f}, {search_end:.3f})")
    print(f"Chosen target accuracy: {best['target_accuracy']:.6f}")
    print(f"Ours hit time: {best['ours_hit_time']:.3f}")
    print(f"Ours train time: {best['ours_train_time']:.3f}")
    print(f"Earliest baseline hit time: {best['min_baseline_hit_time']:.3f}")
    print(f"Earliest baseline train time: {best['min_baseline_train_time']:.3f}")
    print(f"Gap score (baseline - ours): {best['gap']:.3f}")
    print(f"Ratio score (baseline / ours): {best['ratio']:.3f}")
    print("")
    header = (
        f"{'Method':<12}"
        f"{'BestInRange':>14}"
        f"{'HitTime':>12}"
        f"{'TrainTime':>12}"
        f"{'Reached':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in best["rows"]:
        best_in_window = "n/a" if row["best_in_window"] is None else f"{row['best_in_window']:.4f}"
        hit_time = "n/a" if row["hit_time"] is None else f"{row['hit_time']:.3f}"
        train_time = "n/a" if row["time_from_window_start"] is None else f"{row['time_from_window_start']:.3f}"
        reached = "yes" if row["reached"] else "no"
        print(f"{row['method']:<12}{best_in_window:>14}{hit_time:>12}{train_time:>12}{reached:>10}")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "window_start",
                "window_end",
                "target_accuracy",
                "best_in_window",
                "hit_time",
                "time_from_window_start",
                "reached",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--dicg-run-dir", type=Path, default=None)
    parser.add_argument("--mat-run-dir", type=Path, default=None)
    parser.add_argument("--tgcnet-run-dir", type=Path, default=None)
    parser.add_argument("--maple-run-dir", type=Path, default=None)
    parser.add_argument("--comatrack-run-dir", type=Path, default=None)
    parser.add_argument("--maporl-run-dir", type=Path, default=None)
    parser.add_argument("--mpdf-run-dir", type=Path, default=None)
    parser.add_argument("--magrpo-run-dir", type=Path, default=None)
    parser.add_argument("--tag", type=str, default="eval/success_once")
    parser.add_argument("--x-axis", type=str, default="time", choices=["time", "step"])
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--window-end", type=float, default=None)
    parser.add_argument("--window-index", type=int, default=None)
    parser.add_argument("--auto-search-target", action="store_true")
    parser.add_argument("--search-start", type=float, default=None)
    parser.add_argument("--search-end", type=float, default=None)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=1.0)
    parser.add_argument("--unreached-extra-time", type=float, default=1.0)
    parser.add_argument("--min-ours-train-time", type=float, default=1.0)
    parser.add_argument("--require-all-reached", action="store_true")
    parser.add_argument("--search-objective", type=str, default="ratio", choices=["ratio", "gap", "hybrid"])
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument(
        "--segment-time-points",
        type=str,
        default=str(DEFAULT_SEGMENT_POINTS),
    )
    parser.add_argument("--target-mode", type=str, default="topk_mean", choices=["max", "mean", "topk_mean", "quantile"])
    parser.add_argument("--target-topk", type=int, default=3)
    parser.add_argument("--target-quantile", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-stretch-x", dest="stretch_x", action="store_false")
    args = parser.parse_args()

    args.segment_time_points = parse_points(args.segment_time_points)
    method_dirs = build_method_map(args)

    curves = {}
    for method, run_dir in method_dirs.items():
        if run_dir is None:
            continue
        curves[method] = load_curve(Path(run_dir) / "tb", args)

    if args.auto_search_target:
        search_start, search_end = resolve_search_range(args, curves)
        ours_window_xs, ours_window_ys = select_window(
            curves["Ours"]["xs"],
            curves["Ours"]["ys"],
            search_start,
            search_end,
        )
        candidates = build_candidate_thresholds(
            ours_window_ys,
            step=args.threshold_step,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
        )
        if not candidates:
            raise ValueError("No candidate thresholds found in the requested search range.")

        unreached_time = float(search_end + args.unreached_extra_time)
        scored = []
        for target in candidates:
            result = score_candidate(
                curves,
                search_start=search_start,
                search_end=search_end,
                target=target,
                eps=args.eps,
                min_consecutive=args.min_consecutive,
                unreached_time=unreached_time,
                min_ours_train_time=args.min_ours_train_time,
                require_all_reached=args.require_all_reached,
            )
            if result is not None:
                scored.append(result)
        if not scored:
            raise ValueError("No valid target threshold found in auto-search mode.")

        if args.search_objective == "ratio":
            key_fn = lambda item: (
                item["ratio"],
                item["gap"],
                item["target_accuracy"],
                -item["ours_hit_time"],
            )
        elif args.search_objective == "gap":
            key_fn = lambda item: (
                item["gap"],
                item["ratio"],
                item["target_accuracy"],
                -item["ours_hit_time"],
            )
        else:
            key_fn = lambda item: (
                item["ratio"] * item["gap"],
                item["ratio"],
                item["gap"],
                item["target_accuracy"],
                -item["ours_hit_time"],
            )
        best = max(scored, key=key_fn)
        print_search_report(best, search_start, search_end)

        if args.csv is not None:
            write_csv(args.csv, best["rows"])
            print(args.csv)
        if args.json is not None:
            payload = {
                "search_start": float(search_start),
                "search_end": float(search_end),
                "target_accuracy": float(best["target_accuracy"]),
                "gap": float(best["gap"]),
                "ratio": float(best["ratio"]),
                "ours_hit_time": float(best["ours_hit_time"]),
                "ours_train_time": float(best["ours_train_time"]),
                "min_baseline_hit_time": float(best["min_baseline_hit_time"]),
                "min_baseline_train_time": float(best["min_baseline_train_time"]),
                "rows": best["rows"],
            }
            write_json(args.json, payload)
            print(args.json)
        return

    window_start, window_end = resolve_window(args)
    ours_curve = curves["Ours"]
    _, ours_window_ys = select_window(ours_curve["xs"], ours_curve["ys"], window_start, window_end)
    target_accuracy = compute_target(
        ours_window_ys,
        mode=args.target_mode,
        topk=args.target_topk,
        quantile=args.target_quantile,
    )

    rows = []
    for method, curve in curves.items():
        _, window_ys = select_window(curve["xs"], curve["ys"], window_start, window_end)
        best_in_window = None if not window_ys else float(np.max(np.asarray(window_ys, dtype=np.float64)))
        hit = find_first_hit(
            curve["xs"],
            curve["ys"],
            start=window_start,
            end=window_end,
            target=target_accuracy,
            eps=args.eps,
            min_consecutive=args.min_consecutive,
        )
        rows.append(
            {
                "method": method,
                "window_start": float(window_start),
                "window_end": float(window_end),
                "target_accuracy": float(target_accuracy),
                "best_in_window": best_in_window,
                "hit_time": None if hit is None else hit["hit_time"],
                "time_from_window_start": None if hit is None else hit["time_from_window_start"],
                "reached": bool(hit is not None),
            }
        )

    print_report(rows, window_start, window_end, target_accuracy, args.target_mode)

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(args.csv)
    if args.json is not None:
        payload = {
            "window_start": float(window_start),
            "window_end": float(window_end),
            "target_accuracy": float(target_accuracy),
            "target_mode": args.target_mode,
            "rows": rows,
        }
        write_json(args.json, payload)
        print(args.json)


if __name__ == "__main__":
    main()
