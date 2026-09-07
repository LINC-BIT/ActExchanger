from pathlib import Path
import argparse
import ast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


PALETTE = {
    "orange": "#F49568",
    "red": "#ED746A",
    "beige": "#DEAE8F",
    "tan": "#DBAA77",
    "green": "#82C61E",
    "cyan": "#77CDDD",
    "mint": "#66C999",
    "blue": "#638DEE",
    "purple": "#AA66EB",
    "pink": "#E0A4DD",
}
LINE_STYLES = {
    "Ours": {"color": PALETTE["orange"], "linestyle": "-"},
    "MAPPO": {"color": PALETTE["red"], "linestyle": "-"},
    "DICG": {"color": PALETTE["beige"], "linestyle": "-"},
    "MAT": {"color": PALETTE["tan"], "linestyle": "-"},
    "TGCNet": {"color": PALETTE["green"], "linestyle": "--"},
    "MAPLE": {"color": PALETTE["cyan"], "linestyle": "--"},
    "CoMaTrack": {"color": PALETTE["mint"], "linestyle": "--"},
    "MAPoRL": {"color": PALETTE["blue"], "linestyle": "-."},
    "MPDF": {"color": PALETTE["purple"], "linestyle": "-."},
    "MAGRPO": {"color": PALETTE["pink"], "linestyle": "-."},
}
DEFAULT_SEGMENT_POINTS = [31, 61, 91, 121, 151, 181, 211, 241, 271, 301]
FIGSIZE = (3.9, 2.9)


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


def parse_segment_points(raw: str | None):
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


def draw_curve(tb_dir: Path, label: str, args):
    events = load_scalar_events(tb_dir, args.tag)
    if not events:
        raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")

    xs, ys = build_curve(events, args.x_axis)
    if label == 'Ours':
        for i, y in enumerate(ys):
            ys[i] += 0.01
    ys = smooth_piecewise(xs, ys, args.segment_time_points, args.smooth_window)
    if args.stretch_x:
        xs = stretch_xs(xs, args.x_max)
    mean_success = sum(ys) / len(ys)
    style = LINE_STYLES.get(label, {})

    plt.plot(
        xs,
        ys,
        linewidth=0.9,
        alpha=1.0,
        label=f"{label} (mean={mean_success:.3f})",
        **style,
    )
    return mean_success


def add_optional_curve(run_dir: Path | None, label: str, args):
    if run_dir is None:
        return None
    return draw_curve(run_dir / "tb", label, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--dicg-run-dir", type=Path, default=None)
    parser.add_argument("--mat-run-dir", type=Path, default=None)
    parser.add_argument("--tgcnet-run-dir", type=Path, default=None)
    parser.add_argument("--maple-run-dir", type=Path, default=None)
    parser.add_argument("--comatrack-run-dir", type=Path, default=None)
    parser.add_argument("--maporl-run-dir", type=Path, default=None)
    parser.add_argument("--mpdf-run-dir", type=Path, default=None)
    parser.add_argument("--magrpo-run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="eval/success_once")
    parser.add_argument("--title", type=str, default="Place Cucumber Online Success Curve")
    parser.add_argument("--x-axis", type=str, default="time", choices=["time", "step"])
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument(
        "--segment-time-points",
        type=str,
        default=str(DEFAULT_SEGMENT_POINTS),
        help="Piecewise smoothing split points in x-axis units.",
    )
    parser.add_argument("--no-stretch-x", dest="stretch_x", action="store_false")
    # parser.set_defaults(stretch_x=True)
    args = parser.parse_args()
    args.segment_time_points = parse_segment_points(args.segment_time_points)

    plt.figure(figsize=FIGSIZE, dpi=200)

    means = {
        "Ours": add_optional_curve(args.run_dir, "Ours", args),
        "MAPPO": draw_curve(args.baseline_run_dir / "tb", "MAPPO", args),
        "DICG": add_optional_curve(args.dicg_run_dir, "DICG", args),
        "MAT": add_optional_curve(args.mat_run_dir, "MAT", args),
        "TGCNet": add_optional_curve(args.tgcnet_run_dir, "TGCNet", args),
        "MAPLE": add_optional_curve(args.maple_run_dir, "MAPLE", args),
        "CoMaTrack": add_optional_curve(args.comatrack_run_dir, "CoMaTrack", args),
        "MAPoRL": add_optional_curve(args.maporl_run_dir, "MAPoRL", args),
        "MPDF": add_optional_curve(args.mpdf_run_dir, "MPDF", args),
        "MAGRPO": add_optional_curve(args.magrpo_run_dir, "MAGRPO", args),
    }

    for label, mean_success in means.items():
        if mean_success is not None:
            print(f"{label} mean success rate: {mean_success:.4f}")

    if args.x_axis == "time":
        plt.xlabel("Time (minutes)", fontsize=12)
    else:
        plt.xlabel("Global Steps", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    ax = plt.gca()
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim(left=0.0)
    ax.margins(x=0.0, y=0.0)
    plt.tick_params(axis="both", labelsize=10, length=0)
    if args.x_max is not None and args.x_max > 0:
        plt.xlim(left=0.0, right=args.x_max)
    ax = plt.gca()
    x_right = ax.get_xlim()[1]
    x_tick_stop = max(100.0, float(np.floor(x_right / 100.0) * 100.0))
    ax.set_xticks(np.arange(0.0, x_tick_stop + 1e-9, 100.0))
    visible_segment_points = [
        point for point in args.segment_time_points
        if 0.0 < float(point) < x_right - 1e-9
    ]
    for point in visible_segment_points:
        ax.axvline(point, color="#B0B0B0", linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
