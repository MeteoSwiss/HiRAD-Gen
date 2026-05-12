"""Sigma evaluator — sigma curve plots."""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate sigma curve plots from sweep results."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Sigma plot: results_dir=%s, plots_dir=%s", results_dir, plots_dir)
    return plots_dir
