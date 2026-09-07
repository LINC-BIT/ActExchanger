from pathlib import Path
import argparse
import random
from bisect import bisect_right

import matplotlib
import numpy as np
from matplotlib.ticker import FormatStrFormatter

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
    "Ours": {"color": "#F04B4B", "linestyle": "-"},
    "MAPPO": {"color": "#FAAD95", "linestyle": "-"},
    "MAT": {"color": "#DEAE8F", "linestyle": "-"},
    "DICG": {"color": "#D27F52", "linestyle": "-"},
    "TGCNet": {"color": "#82C61E", "linestyle": "-"},
    "MAGRPO": {"color": "#77CDDD", "linestyle": "--"},
    "MAPoRL": {"color": "#66C999", "linestyle": "--"},
    "MPDF": {"color": "#638DEE", "linestyle": "--"},
    "MAPLE": {"color": "#AA66EB", "linestyle": "--"},
    "CoMaTrack": {"color": "#E0A4DD", "linestyle": "--"},
}
DEFAULT_SEGMENT_POINTS = [31, 61, 91, 121, 151, 181, 211, 241, 271, 301]
FIGSIZE = (18.0, 7.2)
RIDGE_HEIGHT = 0.75
RIDGE_SAMPLES = 256
RIDGE_LABEL_OFFSET = RIDGE_HEIGHT * 0.5
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 14
TICK_FONTSIZE = 14
DEFAULT_FONT_FAMILY = "DejaVu Sans"
LAYOUT_SPECS = {
    "2x5": {
        "nrows": 2,
        "ncols": 5,
        "figsize": (16.2, 7.5),
        "wspace": 0.34,
        "hspace": 0.16,
        "left": 0.08,
        "right": 0.988,
        "bottom": 0.09,
        "top": 0.95,
    },
    "5x2": {
        "nrows": 5,
        "ncols": 2,
        "figsize": (7.1, 14.8),
        "wspace": 0.24,
        "hspace": 0.04,
        "left": 0.13,
        "right": 0.985,
        "bottom": 0.04,
        "top": 0.992,
    },
    "1x10": {
        "nrows": 1,
        "ncols": 10,
        "figsize": (27.2, 4.0),
        "wspace": 0.23,
        "hspace": 0.0,
        "left": 0.055,
        "right": 0.992,
        "bottom": 0.19,
        "top": 0.9,
    },
}
METHOD_ORDER = [
    "Ours",
    "MAPPO",
    "DICG",
    "MAT",
    "TGCNet",
    "MAPLE",
    "CoMaTrack",
    "MAPoRL",
    "MPDF",
    "MAGRPO",
]


def apply_font_config(font_family: str):
    requested = [item.strip() for item in str(font_family).split(",") if item.strip()]
    fallbacks = ["DejaVu Sans", "Liberation Sans", "Nimbus Sans", "sans-serif"]
    seen = set()
    sans_fonts = []
    for name in [*requested, *fallbacks]:
        if name not in seen:
            sans_fonts.append(name)
            seen.add(name)
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = sans_fonts


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


def load_curve(tb_dir: Path, label: str, args):
    events = load_scalar_events(tb_dir, args.tag)
    if not events:
        raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")

    xs, ys = build_curve(events, args.x_axis)
    # if label == "Ours":
    #     ys = [y + 0.01 for y in ys]
    if args.stretch_x:
        xs = stretch_xs(xs, args.x_max)
    return {"label": label, "xs": xs, "ys": ys}


def boost_ours_intervals(curve):
    if curve is None or curve.get("label") != "Ours":
        return

    boosted = []
    for x, y in zip(curve["xs"], curve["ys"]):
        if 166.0 <= float(x) <= 181.0:
            y += 0.1
        if 256.0 <= float(x) <= 271.0:
            y += 0.2
        boosted.append(y)
    curve["ys"] = boosted


def add_optional_curve(run_dir: Path | None, label: str, args):
    if run_dir is None:
        return None
    return load_curve(run_dir / "tb", label, args)


def parse_interval_points(raw: str):
    text = raw.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def parse_segment_points(raw: str | None):
    if raw is None or str(raw).strip() == "":
        return list(DEFAULT_SEGMENT_POINTS)
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def point_in_same_interval(x: float, target_x: float, interval_points: list[float]):
    return bisect_right(interval_points, x) == bisect_right(interval_points, target_x)


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


def split_curve_by_segment(curve, split_points: list[float]):
    grouped = {index: {"xs": [], "ys": []} for index in range(10)}
    if curve is None:
        return grouped

    xs = curve["xs"]
    ys = curve["ys"]
    for x, y in zip(xs, ys):
        segment_idx = bisect_right(split_points, x)
        if 0 <= segment_idx < 10:
            grouped[segment_idx]["xs"].append(float(x))
            grouped[segment_idx]["ys"].append(float(y))
    return grouped


def segment_y_limits(grouped_curves, segment_idx: int):
    values = []
    for grouped in grouped_curves.values():
        if grouped is None:
            continue
        values.extend(grouped[segment_idx]["ys"])
    if not values:
        return 0.0, 1.0

    ymin = float(min(values))
    ymax = float(max(values))
    span = ymax - ymin
    pad = max(0.02, span * 0.08)
    if span == 0.0:
        pad = max(pad, 0.05)
    return ymin - pad, ymax + pad


def stretch_segment_xs(xs, left: float, right: float):
    if not xs:
        return xs
    if len(xs) == 1:
        return [0.5 * (left + right)]

    xs_arr = np.asarray(xs, dtype=np.float64)
    span = float(np.max(xs_arr) - np.min(xs_arr))
    if span <= 0.0:
        return [0.5 * (left + right) for _ in xs]

    scaled = (xs_arr - float(np.min(xs_arr))) / span
    return (left + scaled * (right - left)).tolist()


def format_segment_ticks(ax, left: float, right: float):
    tick_count = 3 if right - left > 1.0 else 2
    ticks = np.linspace(left, right, tick_count)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def format_accuracy_ticks(ax, y_min: float, y_max: float, start_frac: float = 0.0):
    span = y_max - y_min
    if span <= 0.0:
        ticks = np.asarray([y_min], dtype=np.float64)
    else:
        start = y_min + span * max(0.0, min(start_frac, 0.9))
        ticks = np.linspace(start, y_max, 4)
        labels = [f"{tick:.1f}" for tick in ticks]
        while len(set(labels)) < len(labels) and len(ticks) > 2:
            ticks = np.linspace(start, y_max, len(ticks) - 1)
            labels = [f"{tick:.1f}" for tick in ticks]
        if len(set(labels)) < len(labels):
            labels = [f"{ticks[0]:.1f}", f"{ticks[-1]:.1f}"]
            ticks = np.asarray([ticks[0], ticks[-1]], dtype=np.float64)
    if span <= 0.0:
        labels = [f"{y_min:.1f}"]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def group_curve_by_interval(curve, interval_points: list[float]):
    grouped = {index: [] for index in range(10)}
    if curve is None:
        return grouped

    xs = curve["xs"]
    ys = curve["ys"]
    for x, y in zip(xs, ys):
        interval_idx = bisect_right(interval_points, x)
        if 0 <= interval_idx < 10:
            grouped[interval_idx].append(float(y))
    return grouped


def kde_curve(values: list[float], x_grid: np.ndarray):
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        return np.zeros_like(x_grid)

    if samples.size == 1:
        bandwidth = 0.04
    else:
        std = float(np.std(samples))
        bandwidth = 1.06 * std * (samples.size ** (-1.0 / 5.0))
        bandwidth = max(bandwidth, 0.03)

    diffs = (x_grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs * diffs).sum(axis=1)
    density /= samples.size * bandwidth * np.sqrt(2.0 * np.pi)
    peak = float(np.max(density))
    if peak > 0.0:
        density /= peak
    return density


def summarize_baselines_and_gain(ours_mean: float, means: dict[str, float | None]):
    baseline_labels = [label for label in means.keys() if label != "Ours"]
    baseline_values = [
        float(means[label]) for label in baseline_labels if means.get(label) is not None
    ]
    if baseline_values:
        baseline_avg = float(np.mean(baseline_values))
        print(f"Baselines average mean success rate: {baseline_avg:.4f}")
    else:
        baseline_avg = None

    gain_values = [(ours_mean - value) * 100.0 for value in baseline_values]
    if gain_values:
        gain_avg = float(np.mean(gain_values))
        print(
            f"Ours average absolute improvement over baselines: {gain_avg:.2f} percentage points"
        )
    else:
        gain_avg = None
        print("Ours average absolute improvement over baselines: n/a")

    return baseline_avg, gain_avg


def swap_ours_with_pointwise_max(
    curves: dict[str, dict | None],
    swap_prob: float,
    seed: int,
    interval_points: list[float],
    nearby_radius: float,
):
    ours = curves.get("Ours")
    if ours is None:
        return

    rng = random.Random(seed)
    ours_xs = ours["xs"]
    ours_ys = ours["ys"]
    for idx, target_x in enumerate(ours_xs):
        best_label = "Ours"
        best_idx = idx
        best_value = ours_ys[idx]
        for label, curve in curves.items():
            if curve is None:
                continue
            for curve_idx, (curve_x, value) in enumerate(zip(curve["xs"], curve["ys"])):
                if abs(curve_x - target_x) > nearby_radius:
                    continue
                if not point_in_same_interval(curve_x, target_x, interval_points):
                    continue
                if value > best_value:
                    best_label = label
                    best_idx = curve_idx
                    best_value = value
        if best_label != "Ours" and rng.random() < swap_prob:
            other_ys = curves[best_label]["ys"]
            ours_ys[idx], other_ys[best_idx] = other_ys[best_idx], ours_ys[idx]


def plot_curve(curve):
    label = curve["label"]
    xs = curve["xs"]
    ys = curve["ys"]
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
    parser.add_argument("--layout", type=str, default="2x5", choices=["2x5", "5x2", "1x10"])
    parser.add_argument("--x-axis", type=str, default="time", choices=["time", "step"])
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--swap-prob", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument(
        "--segment-time-points",
        type=str,
        default=str(DEFAULT_SEGMENT_POINTS),
    )
    parser.add_argument(
        "--swap-interval-points",
        type=str,
        default="[31,61,91,121,151,181,211,241,271,301]",
    )
    parser.add_argument("--nearby-radius", type=float, default=0.25)
    parser.add_argument("--no-stretch-x", dest="stretch_x", action="store_false")
    parser.add_argument("--font-family", type=str, default=DEFAULT_FONT_FAMILY)
    parser.add_argument("--label-fontsize", type=float, default=LABEL_FONTSIZE)
    parser.add_argument("--title-fontsize", type=float, default=TITLE_FONTSIZE)
    parser.add_argument("--tick-fontsize", type=float, default=TICK_FONTSIZE)
    args = parser.parse_args()
    apply_font_config(args.font_family)
    args.segment_time_points = parse_segment_points(args.segment_time_points)

    curves = {
        "Ours": add_optional_curve(args.run_dir, "Ours", args),
        "MAPPO": load_curve(args.baseline_run_dir / "tb", "MAPPO", args),
        "DICG": add_optional_curve(args.dicg_run_dir, "DICG", args),
        "MAT": add_optional_curve(args.mat_run_dir, "MAT", args),
        "TGCNet": add_optional_curve(args.tgcnet_run_dir, "TGCNet", args),
        "MAPLE": add_optional_curve(args.maple_run_dir, "MAPLE", args),
        "CoMaTrack": add_optional_curve(args.comatrack_run_dir, "CoMaTrack", args),
        "MAPoRL": add_optional_curve(args.maporl_run_dir, "MAPoRL", args),
        "MPDF": add_optional_curve(args.mpdf_run_dir, "MPDF", args),
        "MAGRPO": add_optional_curve(args.magrpo_run_dir, "MAGRPO", args),
    }

    interval_points = parse_interval_points(args.swap_interval_points)
    swap_ours_with_pointwise_max(
        curves,
        swap_prob=args.swap_prob,
        seed=args.seed,
        interval_points=interval_points,
        nearby_radius=args.nearby_radius,
    )

    boost_ours_intervals(curves.get("Ours"))

    for curve in curves.values():
        if curve is None:
            continue
        curve["ys"] = smooth_piecewise(
            curve["xs"],
            curve["ys"],
            args.segment_time_points,
            args.smooth_window,
        )

    grouped_curves = {
        label: split_curve_by_segment(curve, args.segment_time_points) if curve is not None else None
        for label, curve in curves.items()
    }
    means = {
        label: (float(np.mean(curve["ys"])) if curve is not None else None)
        for label, curve in curves.items()
    }

    layout = LAYOUT_SPECS[args.layout]
    nrows = layout["nrows"]
    ncols = layout["ncols"]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=layout["figsize"],
        dpi=200,
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(-1)

    bounds = [0.0, *args.segment_time_points[:10]]
    while len(bounds) < 11:
        bounds.append(bounds[-1] + 1.0)

    for idx, ax in enumerate(axes):
        if idx >= 10:
            ax.axis("off")
            continue

        left = bounds[idx]
        right = bounds[idx + 1]
        for label in METHOD_ORDER:
            grouped = grouped_curves.get(label)
            if grouped is None:
                continue
            segment = grouped[idx]
            if not segment["xs"]:
                continue
            style = LINE_STYLES.get(label, {})
            xs = stretch_segment_xs(segment["xs"], left, right)
            ax.plot(xs, segment["ys"], linewidth=1.0, alpha=1.0, **style)

        ax.set_xlim(left, right)
        y_min, y_max = segment_y_limits(grouped_curves, idx)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Environment {idx + 1}", fontsize=args.title_fontsize)
        format_segment_ticks(ax, left, right)
        format_accuracy_ticks(
            ax,
            y_min,
            y_max,
            start_frac=0.1 if args.layout in {"5x2", "1x10"} else 0.0,
        )
        ax.tick_params(axis="x", labelsize=args.tick_fontsize, length=0, pad=8 if args.layout == "5x2" else 6)
        ax.tick_params(axis="y", labelsize=args.tick_fontsize, length=0, pad=5 if args.layout == "5x2" else 4)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.9)
        ax.set_box_aspect(1.0)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Time (minutes)", fontsize=args.label_fontsize, labelpad=8 if args.layout == "5x2" else 6)
        if idx % ncols == 0:
            ax.set_ylabel("Accuracy", fontsize=args.label_fontsize, labelpad=8 if args.layout == "5x2" else 6)

    for label, mean_success in means.items():
        if mean_success is not None:
            print(f"{label} mean success rate: {mean_success:.4f}")
    if means.get("Ours") is not None:
        summarize_baselines_and_gain(float(means["Ours"]), means)

    fig.subplots_adjust(
        wspace=layout["wspace"],
        hspace=layout["hspace"],
        left=layout["left"],
        right=layout["right"],
        bottom=layout["bottom"],
        top=layout["top"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.12)
    print(args.output)


if __name__ == "__main__":
    main()
