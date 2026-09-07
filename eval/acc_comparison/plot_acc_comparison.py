#!/usr/bin/env python3
"""Plot accuracy comparisons for all four multi-agent workloads."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


WORKLOADS = (
    "object_picking_placing",
    "object_stacking",
    "cucumber_placing",
    "cylinder_transfer",
)
RUN_VARIABLES = (
    "RUN_DIR",
    "BASELINE_RUN_DIR",
    "DICG_RUN_DIR",
    "MAT_RUN_DIR",
    "TGCNET_RUN_DIR",
    "MAPLE_RUN_DIR",
    "COMATRACK_RUN_DIR",
    "MAPORL_RUN_DIR",
    "MPDF_RUN_DIR",
    "MAGRPO_RUN_DIR",
)


def workload_environment(workload: str) -> dict[str, str]:
    environment = os.environ.copy()
    prefix = workload.upper()
    for name in RUN_VARIABLES:
        workload_name = f"{prefix}_{name}"
        if workload_name in environment:
            environment[name] = environment[workload_name]
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    for workload in WORKLOADS:
        wrapper = script_dir / workload / "run_acc_comparison.sh"
        print(f"[plot_acc_comparison] plotting {workload}")
        subprocess.run(
            ["bash", str(wrapper)],
            check=True,
            cwd=script_dir.parent.parent,
            env=workload_environment(workload),
        )


if __name__ == "__main__":
    main()
