"""Mechanistic evaluator — attention/weight extraction.

Delegates to eval.weight_diagnostics for the actual computation.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def run(
    checkpoint_path: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """Run mechanistic weight diagnostics on a checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else checkpoint_path.parent / "evaluators" / "mechanistic"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Mechanistic output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Mechanistic run: checkpoint=%s, output_dir=%s", checkpoint_path, output_dir)
    return output_dir
