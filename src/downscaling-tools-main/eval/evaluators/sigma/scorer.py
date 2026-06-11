"""Sigma evaluator — cross-schedule scoring."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    """Score sigma sweep results."""
    LOG.info("Sigma scorer: results_dir=%s", results_dir)
    return []
