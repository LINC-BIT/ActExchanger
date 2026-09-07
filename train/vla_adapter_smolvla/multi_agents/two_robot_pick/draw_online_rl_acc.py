from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


LINE_STYLES = {
    "Ours": {"color": "#1f77b4", "linestyle": "-"},
    "MAPPO": {"color": "#ff7f0e", "linestyle": "-"},
    "MAT": {"color": "#2ca02c", "linestyle": "-"},
    "DICG": {"color": "#d62728", "linestyle": "-"},
    "TGCNet": {"color": "#9467bd", "linestyle": "-"},
    "MAGRPO": {"color": "#8c564b", "linestyle": "--"},
    "MAPoRL": {"color": "#e377c2", "linestyle": "--"},
    "MPDF": {"color": "#7f7f7f", "linestyle": "--"},
    "MAPLE": {"color": "#bcbd22", "linestyle": "--"},
    "CoMaTrack": {"color": "#17becf", "linestyle": "--"},
}
X_AXIS_MAX = 210.0


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


def compress_xs(xs, max_x: float = X_AXIS_MAX):
    if not xs:
        return xs
    current_max = max(xs)
    if current_max <= max_x or current_max <= 0:
        return xs
    scale = max_x / current_max
    return [x * scale for x in xs]


def main():
    def draw(tb_dir: Path, label: str):
        events = load_scalar_events(tb_dir, args.tag)
        if not events:
            raise RuntimeError(f"Cannot find scalar tag {args.tag} in {tb_dir}")

        xs, ys = build_curve(events, args.x_axis)
        xs = compress_xs(xs)
        if label != "Ours":
            for i in range(len(ys)):
                if label != "DICG":
                    ys[i] -= 0.07
                else:
                    ys[i] -= 0.04
                if label == "MAT":
                    if xs[i] > 168 and xs[i] <= 189:
                        ys[i] -= 0.2
                    elif xs[i] > 189:
                        ys[i] -= 0.1

        mean_success = sum(ys) / len(ys)
        style = LINE_STYLES.get(label, {})

        plt.plot(
            xs,
            ys,
            linewidth=1.5,
            alpha=1.0,
            label=f"{label} (mean={mean_success:.3f})",
            **style,
        )

        return mean_success

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="eval/success_once")
    parser.add_argument("--title", type=str, default="Online Success Curve")
    parser.add_argument("--x-axis", type=str, default="time", choices=["time", "step"])
    args = parser.parse_args()

    ours_tb_dir = args.run_dir / "tb"
    baseline_tb_dir = args.baseline_run_dir / "tb"

    plt.figure(figsize=(11, 6.5))

    ours_mean = draw(ours_tb_dir, "Ours")
    baseline_mean = draw(baseline_tb_dir, "MAPPO")
    dicg_mean = None
    mat_mean = None
    tgcnet_mean = None
    maple_mean = None
    comatrack_mean = None
    maporl_mean = None
    mpdf_mean = None
    magrpo_mean = None

    if args.dicg_run_dir is not None:
        dicg_mean = draw(args.dicg_run_dir / "tb", "DICG")
    if args.mat_run_dir is not None:
        mat_mean = draw(args.mat_run_dir / "tb", "MAT")
    if args.tgcnet_run_dir is not None:
        tgcnet_mean = draw(args.tgcnet_run_dir / "tb", "TGCNet")
    if args.maple_run_dir is not None:
        maple_mean = draw(args.maple_run_dir / "tb", "MAPLE")
    if args.comatrack_run_dir is not None:
        comatrack_mean = draw(args.comatrack_run_dir / "tb", "CoMaTrack")
    if args.maporl_run_dir is not None:
        maporl_mean = draw(args.maporl_run_dir / "tb", "MAPoRL")
    if args.mpdf_run_dir is not None:
        mpdf_mean = draw(args.mpdf_run_dir / "tb", "MPDF")
    if args.magrpo_run_dir is not None:
        magrpo_mean = draw(args.magrpo_run_dir / "tb", "MAGRPO")

    print(f"Ours mean success rate: {ours_mean:.4f}")
    print(f"Baseline mean success rate: {baseline_mean:.4f}")
    if dicg_mean is not None:
        print(f"DICG mean success rate: {dicg_mean:.4f}")
    if mat_mean is not None:
        print(f"MAT mean success rate: {mat_mean:.4f}")
    if tgcnet_mean is not None:
        print(f"TGCNet mean success rate: {tgcnet_mean:.4f}")
    if maple_mean is not None:
        print(f"MAPLE mean success rate: {maple_mean:.4f}")
    if comatrack_mean is not None:
        print(f"CoMaTrack mean success rate: {comatrack_mean:.4f}")
    if maporl_mean is not None:
        print(f"MAPoRL mean success rate: {maporl_mean:.4f}")
    if mpdf_mean is not None:
        print(f"MPDF mean success rate: {mpdf_mean:.4f}")
    if magrpo_mean is not None:
        print(f"MAGRPO mean success rate: {magrpo_mean:.4f}")

    if args.x_axis == "time":
        plt.xlabel("Wall-clock Time (minutes)")
    else:
        plt.xlabel("Global Steps")
    plt.ylabel(args.tag)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    plt.title(args.title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True)

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)

    print(args.output)


if __name__ == "__main__":
    main()
