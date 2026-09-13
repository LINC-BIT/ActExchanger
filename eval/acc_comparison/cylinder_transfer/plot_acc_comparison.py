from pathlib import Path
import argparse
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


LINE_STYLES = {
    "Ours": {"color": "#d62728", "linestyle": "-"},
    "MAPPO": {"color": "#ff7f0e", "linestyle": "-"},
    "DICG": {"color": "#2ca02c", "linestyle": "-"},
    "MAT": {"color": "#1f77b4", "linestyle": "-"},
    "TGCNet": {"color": "#9467bd", "linestyle": "--"},
    "MAPLE": {"color": "#8c564b", "linestyle": "--"},
    "CoMaTrack": {"color": "#e377c2", "linestyle": "--"},
    "MAPoRL": {"color": "#7f7f7f", "linestyle": "-."},
    "MPDF": {"color": "#bcbd22", "linestyle": "-."},
    "MAGRPO": {"color": "#17becf", "linestyle": "-."},
}


def load_scalar_events(tb_dir: Path, tag: str):
    if tb_dir.exists():
        accumulator = event_accumulator.EventAccumulator(
            str(tb_dir),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        if tag in accumulator.Tags().get("scalars", []):
            return accumulator.Scalars(tag)

    metrics_path = tb_dir.parent / "metrics.json"
    if not metrics_path.is_file():
        return []
    try:
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []

    points = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        try:
            value = float(row.get("score", row.get("success_once")))
        except (TypeError, ValueError):
            continue
        step = int(row.get("step", index))
        elapsed_minutes = float(row.get("elapsed_minutes", index))
        points.append(SimpleNamespace(step=step, value=value, wall_time=elapsed_minutes * 60.0))
    return points

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


def draw_curve(tb_dir: Path, label: str, args):
    events = load_scalar_events(tb_dir, args.tag)
    if not events:
        raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")

    xs, ys = build_curve(events, args.x_axis)
    if label == 'Ours':
        for i, y in enumerate(ys):
            ys[i] += 0.01
    if args.stretch_x:
        xs = stretch_xs(xs, args.x_max)
    mean_success = sum(ys) / len(ys)
    style = LINE_STYLES.get(label, {})

    plt.plot(
        xs,
        ys,
        linewidth=1.5,
        alpha=0.95,
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
    parser.add_argument("--no-stretch-x", dest="stretch_x", action="store_false")
    # parser.set_defaults(stretch_x=True)
    args = parser.parse_args()

    plt.figure(figsize=(11, 6.5))

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
        plt.xlabel("Wall-clock Time (minutes)")
    else:
        plt.xlabel("Global Steps")
    plt.ylabel(args.tag)
    plt.ylim(0.0, 1.0)
    if args.x_max is not None and args.x_max > 0:
        plt.xlim(left=0.0, right=args.x_max)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    plt.title(args.title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
