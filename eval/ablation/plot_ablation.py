from __future__ import annotations

import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


XLSX_PATH = Path(__file__).resolve().parent / "data/531 ablation.xlsx"
OUT_PNG = Path(__file__).resolve().parents[2] / "tmp/531_ablation.png"
OUT_PDF = Path(__file__).resolve().parents[2] / "tmp/531_ablation.pdf"

CHOICE_LABEL_MAP = {
    "raw features": "raw features",
    "semantic-only features": "semantic features",
    "spatial-only features": "spatial features",
    "semantic-spatial hybrid features": "semantic and spatial features",
    "original length": "original feature length",
    "mdl": "MDL",
    "multiple unfused features per action": "knowledge in multiple vectors",
    "single fused feature per action": "knowledge in a separate vector",
    "no aggregation": "not generated from other features",
    "current-forward aggregation": "generated for the current action",
    "random aggregation": "generated for a random action",
    "same-action aggregation": "generated for the same action",
}

GROUP_DISPLAY = {
    "Constraining action-grained feature length by": "(a) Feature length",
    "Generating action-grained features using": "(b) Feature generation",
    "Representing action-grained features as": "(c) Knowledge representation",
    "Aggregation strategy": "(d) Aggregation strategy",
}

GROUP_ORDER = {
    "Constraining action-grained feature length by": 0,
    "Generating action-grained features using": 1,
    "Representing action-grained features as": 2,
    "Aggregation strategy": 3,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=XLSX_PATH)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--gap", type=float, default=0.30)
    parser.add_argument("--inner", type=float, default=0.40)
    parser.add_argument("--y-pad", type=float, default=0.20)
    parser.add_argument("--bar-h", type=float, default=0.22)
    parser.add_argument("--fig-width", type=float, default=5.92)
    parser.add_argument("--fig-height", type=float, default=4.48)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--bar-panel-width-scale", type=float, default=1.0)
    parser.add_argument("--bar-panel-height-scale", type=float, default=1.0)
    parser.add_argument("--grid-left", type=float, default=0.02)
    parser.add_argument("--grid-right", type=float, default=0.985)
    parser.add_argument("--grid-top", type=float, default=0.992)
    parser.add_argument("--grid-bottom", type=float, default=0.068)
    parser.add_argument("--grid-wspace", type=float, default=0.01)
    parser.add_argument("--blank-ratio", type=float, default=0.08)
    parser.add_argument("--text-ratio", type=float, default=3.06)
    parser.add_argument("--bar-ratio", type=float, default=1.86)
    parser.add_argument("--text-x", type=float, default=0.98)
    parser.add_argument("--text-fontsize", type=float, default=9.7)
    parser.add_argument("--xlabel-fontsize", type=float, default=10.5)
    parser.add_argument("--xtick-fontsize", type=float, default=8.9)
    parser.add_argument("--hatch-linewidth", type=float, default=0.50)
    parser.add_argument("--best-facecolor", default="#d1d1d1")
    parser.add_argument("--best-edgecolor", default="black")
    parser.add_argument("--best-linewidth", type=float, default=0.62)
    parser.add_argument("--best-hatch", default="//")
    parser.add_argument("--normal-facecolor", default="#989898")
    parser.add_argument("--normal-edgecolor", default="black")
    parser.add_argument("--normal-linewidth", type=float, default=0.62)
    parser.add_argument("--tick-color", default="#949494")
    parser.add_argument("--tick-width", type=float, default=0.40)
    parser.add_argument("--spine-color", default="#1f1f1f")
    parser.add_argument("--spine-width", type=float, default=0.72)
    parser.add_argument("--grid-color", default="#e1e1e1")
    parser.add_argument("--grid-width", type=float, default=0.36)
    parser.add_argument("--target-max-acc", type=float, default=0.8027)
    return parser.parse_args()


def load_rows(path: Path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                text = "".join((t.text or "") for t in si.iterfind(".//a:t", ns))
                shared.append(text)

        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            values = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib["r"]
                col = "".join(ch for ch in ref if ch.isalpha())
                cell_type = cell.attrib.get("t")
                v = cell.find("a:v", ns)
                if v is None:
                    values[col] = ""
                    continue
                text = v.text or ""
                if cell_type == "s":
                    values[col] = shared[int(text)]
                else:
                    values[col] = text
            rows.append(values)
    return rows


def build_groups(rows):
    groups = []
    current = None
    for row in rows[2:]:
        type_name = row.get("A", "").strip()
        if type_name:
            current = {"type": type_name, "items": []}
            groups.append(current)
        if current is None:
            continue
        choice = row.get("B", "").strip()
        acc = float(row.get("C", "0") or 0)
        current["items"].append({"choice": choice, "acc": acc})
    groups.sort(key=lambda group: GROUP_ORDER.get(group["type"], len(GROUP_ORDER)))
    return groups


def normalize_choice_label(label: str) -> str:
    if not label:
        return label
    mapped = CHOICE_LABEL_MAP.get(label.strip().lower())
    if mapped is not None:
        return mapped
    if label.isupper():
        return label
    return label[0].lower() + label[1:]


def main():
    args = parse_args()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Nimbus Sans",
        "sans-serif",
    ]
    plt.rcParams["hatch.linewidth"] = args.hatch_linewidth

    rows = load_rows(args.xlsx)
    groups = build_groups(rows)

    y_positions = []
    labels = []
    values = []
    is_best = []
    group_labels = []

    gap = args.gap
    inner = args.inner
    y = 0.0
    for group in groups:
        group_start_y = y
        for idx, item in enumerate(group["items"]):
            y_positions.append(y)
            labels.append(normalize_choice_label(item["choice"]))
            values.append(item["acc"])
            is_best.append(idx == len(group["items"]) - 1)
            y += inner
        group_end_y = y - inner
        group_labels.append((0.5 * (group_start_y + group_end_y), GROUP_DISPLAY.get(group["type"], group["type"])))
        y += gap

    if values:
        current_max = max(values)
        if current_max > 0:
            scale = args.target_max_acc / current_max
            values = [value * scale for value in values]

    fig = plt.figure(
        figsize=(args.fig_width, args.fig_height * args.bar_panel_height_scale),
        dpi=args.dpi,
    )
    gs = GridSpec(
        1,
        3,
        width_ratios=[
            args.blank_ratio,
            args.text_ratio,
            args.bar_ratio * args.bar_panel_width_scale,
        ],
        left=args.grid_left,
        right=args.grid_right,
        top=args.grid_top,
        bottom=args.grid_bottom,
        wspace=args.grid_wspace,
    )

    ax_blank = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_text)

    y_pad = args.y_pad
    ax_blank.set_xlim(0, 1)
    ax_blank.set_ylim(max(y_positions) + y_pad, -y_pad)
    ax_blank.axis("off")
    for group_y, group_label in group_labels:
        ax_blank.text(0.98, group_y, group_label, ha="right", va="center", fontsize=args.text_fontsize, color="black")

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(max(y_positions) + y_pad, -y_pad)
    ax_text.axis("off")
    for yv, label in zip(y_positions, labels):
        ax_text.text(
            args.text_x,
            yv,
            label,
            ha="right",
            va="center",
            fontsize=args.text_fontsize,
            color="black",
            family="Arial",
        )

    bar_h = args.bar_h
    for yv, val, best in zip(y_positions, values, is_best):
        if best:
            ax_bar.barh(
                yv,
                val,
                height=bar_h,
                facecolor=args.best_facecolor,
                edgecolor=args.best_edgecolor,
                hatch=args.best_hatch,
                linewidth=args.best_linewidth,
                zorder=4,
                clip_on=True,
            )
        else:
            ax_bar.barh(
                yv,
                val,
                height=bar_h,
                color=args.normal_facecolor,
                edgecolor=args.normal_edgecolor,
                linewidth=args.normal_linewidth,
                zorder=4,
                clip_on=True,
            )

    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_ylim(max(y_positions) + y_pad, -y_pad)
    ax_bar.set_xlabel("Accuracy", fontsize=args.xlabel_fontsize)
    ax_bar.set_xticks([0.0, 0.5, 1.0])
    ax_bar.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=args.xtick_fontsize)
    ax_bar.set_yticks(y_positions)
    ax_bar.tick_params(axis="y", left=True, labelleft=False, length=1.6, width=args.tick_width, color=args.tick_color)
    ax_bar.tick_params(axis="x", length=0, pad=2.5)
    ax_bar.xaxis.grid(True, which="major", color=args.grid_color, linewidth=args.grid_width, zorder=0)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(args.spine_width)
        spine.set_color(args.spine_color)
        spine.set_zorder(1)

    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    print(args.out_png)
    print(args.out_pdf)


if __name__ == "__main__":
    main()
