"""Surface evaluator — compute and persist surface loss metrics.

Calls the standard process_predictions_dir kernel and writes the result
as surface_loss.json so the scorer can load it without re-reading NC files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval._backends.scoreboard._surface_compute import process_predictions_dir

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
    """Compute surface loss metrics and write surface_loss.json."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "surface_loss.json"
    if json_path.exists() and not overwrite:
        LOG.info("surface_loss.json already exists, skipping: %s", json_path)
        return output_dir

    raw = process_predictions_dir(predictions_dir)
    json_path.write_text(json.dumps(raw, indent=2, default=str) + "\n")
    LOG.info("Surface loss written to %s", json_path)
    return output_dir
