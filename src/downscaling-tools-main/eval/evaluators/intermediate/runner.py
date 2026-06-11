"""Intermediate evaluator — run diffusion and capture intermediate steps.

Delegates to eval.plot_intermediate for the actual computation.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Run intermediate step capture and visualization."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "intermediate"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Intermediate output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Intermediate run: predictions_dir=%s, output_dir=%s", predictions_dir, output_dir)
    return output_dir
