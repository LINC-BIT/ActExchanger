#!/usr/bin/env python3
"""Measure model/checkpoint size and optional CUDA resident memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def measure(path: Path, label: str, device: torch.device, budget_gb: float) -> dict[str, Any]:
    candidates = [path / name for name in ("best_agent.pt", "latest_agent.pt")]
    checkpoint = next((item for item in candidates if item.is_file()), path if path.is_file() else None)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint under {path}")
    file_mb = checkpoint.stat().st_size / 1024**2
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("agent", payload.get("model", payload)) if isinstance(payload, dict) else payload
    tensors = state.values() if isinstance(state, dict) else []
    parameters = sum(int(value.numel()) for value in tensors if torch.is_tensor(value))
    parameter_mb = sum(int(value.numel()) * int(value.element_size()) for value in tensors if torch.is_tensor(value)) / 1024**2
    resident_mb = parameter_mb
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loaded = [value.to(device) for value in tensors if torch.is_tensor(value)]
        torch.cuda.synchronize(device)
        resident_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        del loaded
        torch.cuda.empty_cache()
    supported_gb = (file_mb / 1024.0) * (budget_gb * 1024.0 / resident_mb) if resident_mb else 0.0
    return {
        "workload": label,
        "label": label.replace("_", " ").title(),
        "checkpoint": str(checkpoint),
        "checkpoint_mb": file_mb,
        "parameter_count": parameters,
        "model_size_gb": file_mb / 1024.0,
        "resident_memory_mb": resident_mb,
        "max_supported_model_gb": supported_gb,
        "largest_platform_model_gb": budget_gb,
        "knowledge_mb": 0.0,
        "module_overhead_mb": max(0.0, resident_mb - parameter_mb),
        "budget_gb": budget_gb,
        "device": str(device),
        "source": "checkpoint_measurement",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True, help="LABEL::CHECKPOINT_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budget-gb", type=float, default=32.0)
    args = parser.parse_args()
    rows = []
    for raw in args.entry:
        try:
            label, path = raw.split("::", 1)
        except ValueError as error:
            raise SystemExit(f"invalid --entry: {raw}") from error
        rows.append(measure(Path(path), label, torch.device(args.device), args.budget_gb))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
