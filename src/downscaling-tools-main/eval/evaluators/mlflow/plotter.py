"""MLflow evaluator — plotter (no-op: plots are written by the runner)."""
from __future__ import annotations

from pathlib import Path


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    return Path(output_dir) if output_dir else Path(results_dir)
