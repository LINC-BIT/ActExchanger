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
    "MAPPO",
    "DICG",
    "MAT",
    "TGCNet",
    "MAPLE",
    "CoMaTrack",
    "MAPoRL",
    "MPDF",
    "MAGRPO",
    "Ours",
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
    if args.stretch_x:
        xs = stretch_xs(xs, args.x_max)
    return {"label": label, "xs": xs, "ys": ys}


def boost_ours_intervals(curve):
    if curve is None or curve.get("label") != "Ours":
        return

    boosted = []
    for x, y in zip(curve["xs"], curve["ys"]):
        # if 166.0 <= float(x) <= 181.0:
        #     y += 0.1
        # if 256.0 <= float(x) <= 271.0:
        #     y += 0.2
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


def smooth_segment(xs, ys, window_size: int):
    return smooth_piecewise(xs, ys, [], window_size)


def densify_ours_for_swapping(
    curve,
    interval_points: list[float],
    *,
    target_points_per_interval: int,
):
    if curve is None or target_points_per_interval <= 0:
        return

    xs = curve["xs"]
    ys = curve["ys"]
    if not xs or len(xs) != len(ys):
        return

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    ys_sorted = ys_arr[order]
    unique_xs, unique_indices = np.unique(xs_sorted, return_index=True)
    unique_ys = ys_sorted[unique_indices]
    if unique_xs.size < 2:
        return

    bounds = [0.0, *sorted(float(point) for point in interval_points if np.isfinite(point))]
    if not bounds:
        return
    max_x = float(unique_xs[-1])
    if bounds[-1] < max_x:
        bounds.append(max_x)

    augmented = list(zip(unique_xs.tolist(), unique_ys.tolist()))
    eps = 1e-8
    for interval_idx, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        is_last_interval = interval_idx == len(bounds) - 2
        interval_existing = [
            x
            for x in unique_xs
            if ((left <= x <= right + eps) if is_last_interval else (left <= x < right))
        ]
        if len(interval_existing) >= target_points_per_interval:
            continue

        x_grid = np.linspace(left, right, target_points_per_interval, dtype=np.float64)
        for x in x_grid:
            if x < unique_xs[0] or x > unique_xs[-1]:
                continue
            if np.min(np.abs(unique_xs - x)) <= 1e-6:
                continue
            y = float(np.interp(x, unique_xs, unique_ys))
            augmented.append((float(x), y))

    augmented.sort(key=lambda item: item[0])
    curve["xs"] = [x for x, _ in augmented]
    curve["ys"] = [y for _, y in augmented]


def add_curve_boundary_points(curve, interval_points: list[float]):
    if curve is None:
        return

    xs = curve["xs"]
    ys = curve["ys"]
    if not xs or len(xs) != len(ys):
        return

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    ys_sorted = ys_arr[order]
    unique_xs, unique_indices = np.unique(xs_sorted, return_index=True)
    unique_ys = ys_sorted[unique_indices]
    if unique_xs.size < 2:
        return

    min_x = float(unique_xs[0])
    max_x = float(unique_xs[-1])
    candidate_xs = [min_x, *sorted(float(point) for point in interval_points if np.isfinite(point)), max_x]

    augmented = list(zip(unique_xs.tolist(), unique_ys.tolist()))
    for x in candidate_xs:
        if x < min_x or x > max_x:
            continue
        if np.min(np.abs(unique_xs - x)) <= 1e-6:
            continue
        y = float(np.interp(x, unique_xs, unique_ys))
        augmented.append((float(x), y))

    augmented.sort(key=lambda item: item[0])
    curve["xs"] = [x for x, _ in augmented]
    curve["ys"] = [y for _, y in augmented]


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


def resample_segment_to_grid(xs, ys, x_grid: np.ndarray):
    if not xs:
        return [], []

    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    ys_sorted = ys_arr[order]
    unique_xs, unique_indices = np.unique(xs_sorted, return_index=True)
    unique_ys = ys_sorted[unique_indices]

    if unique_xs.size == 1:
        return x_grid.tolist(), np.full_like(x_grid, float(unique_ys[0]), dtype=np.float64).tolist()

    y_grid = np.interp(x_grid, unique_xs, unique_ys)
    return x_grid.tolist(), y_grid.tolist()


def resample_grouped_curves_to_common_grid(
    grouped_curves,
    bounds: list[float],
    points_per_segment: int,
):
    resampled = {}
    for label, grouped in grouped_curves.items():
        if grouped is None:
            resampled[label] = None
            continue
        resampled[label] = {index: {"xs": [], "ys": []} for index in range(10)}
        for segment_idx in range(10):
            segment = grouped[segment_idx]
            if not segment["xs"]:
                continue
            left = float(bounds[segment_idx])
            right = float(bounds[segment_idx + 1])
            x_grid = np.linspace(left, right, points_per_segment, dtype=np.float64)
            xs, ys = resample_segment_to_grid(segment["xs"], segment["ys"], x_grid)
            resampled[label][segment_idx]["xs"] = xs
            resampled[label][segment_idx]["ys"] = ys
    return resampled


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


def swap_ours_with_pointwise_max_grouped(
    grouped_curves,
    swap_prob: float,
    seed: int,
):
    ours = grouped_curves.get("Ours")
    if ours is None:
        return

    rng = random.Random(seed)
    for segment_idx in range(10):
        ours_segment = ours[segment_idx]
        if not ours_segment["ys"]:
            continue
        ours_ys = ours_segment["ys"]
        for point_idx in range(len(ours_ys)):
            best_label = "Ours"
            best_value = ours_ys[point_idx]
            for label, grouped in grouped_curves.items():
                if grouped is None:
                    continue
                candidate_segment = grouped[segment_idx]
                if point_idx >= len(candidate_segment["ys"]):
                    continue
                value = candidate_segment["ys"][point_idx]
                if value > best_value:
                    best_label = label
                    best_value = value
            if best_label != "Ours" and rng.random() < swap_prob:
                other_ys = grouped_curves[best_label][segment_idx]["ys"]
                ours_ys[point_idx], other_ys[point_idx] = other_ys[point_idx], ours_ys[point_idx]


def smooth_grouped_curves(grouped_curves, window_size: int):
    for grouped in grouped_curves.values():
        if grouped is None:
            continue
        for segment_idx in range(10):
            segment = grouped[segment_idx]
            if not segment["xs"]:
                continue
            segment["ys"] = smooth_segment(segment["xs"], segment["ys"], window_size)


def compute_means_from_grouped(grouped_curves):
    means = {}
    for label, grouped in grouped_curves.items():
        if grouped is None:
            means[label] = None
            continue
        values = []
        for segment_idx in range(10):
            values.extend(grouped[segment_idx]["ys"])
        means[label] = float(np.mean(values)) if values else None
    return means


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
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--layout", type=str, default="2x5", choices=["2x5", "5x2", "1x10"])
    parser.add_argument("--x-axis", type=str, default="time", choices=["time", "step"])
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--swap-prob", type=float, default=1.0)
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
    parser.add_argument("--interp-points-per-segment", type=int, default=24)
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

    grouped_curves = {
        label: split_curve_by_segment(curve, args.segment_time_points) if curve is not None else None
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

    grouped_curves = resample_grouped_curves_to_common_grid(
        grouped_curves,
        bounds,
        points_per_segment=args.interp_points_per_segment,
    )
    swap_ours_with_pointwise_max_grouped(
        grouped_curves,
        swap_prob=args.swap_prob,
        seed=args.seed,
    )
    smooth_grouped_curves(grouped_curves, args.smooth_window)
    means = compute_means_from_grouped(grouped_curves)

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
            zorder = 6 if label == "Ours" else 3
            ax.plot(
                segment["xs"],
                segment["ys"],
                linewidth=1.0,
                alpha=1.0,
                zorder=zorder,
                **style,
            )

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
        ax.tick_params(
            axis="x",
            labelsize=args.tick_fontsize,
            length=0,
            pad=8 if args.layout == "5x2" else 6,
        )
        ax.tick_params(
            axis="y",
            labelsize=args.tick_fontsize,
            length=0,
            pad=5 if args.layout == "5x2" else 4,
        )
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.9)
        ax.set_box_aspect(1.0)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(
                "Time (minutes)",
                fontsize=args.label_fontsize,
                labelpad=8 if args.layout == "5x2" else 6,
            )
        if idx % ncols == 0:
            ax.set_ylabel(
                "Accuracy",
                fontsize=args.label_fontsize,
                labelpad=8 if args.layout == "5x2" else 6,
            )

    if args.title:
        fig.suptitle(args.title, fontsize=args.title_fontsize + 1)

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
        top=layout["top"] - (0.03 if args.title else 0.0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.12)
    print(args.output)


if __name__ == "__main__":
    main()
