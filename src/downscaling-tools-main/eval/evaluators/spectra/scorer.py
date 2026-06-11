"""Spectra evaluator — L2 distance scoring.

Wraps eval.scoreboard.spectra scoring functions with the standard
evaluator interface. Scoring algorithms are imported directly to
guarantee mathematical equivalence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard.spectra import (
    SPECTRA_FIELDS,
    load_spectra_metrics,
    spectra_score,
)

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    reference_root: str | Path | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Score spectra results from a spectra output directory.

    Returns list of {"metric": str, "value": float, "unit": str} records.
    """
    results_dir = Path(results_dir)
    ref_root = Path(reference_root) if reference_root else None

    fields = eval_config.get("fields", list(SPECTRA_FIELDS))
    threshold = eval_config.get("wavenumber_threshold")

    metrics = load_spectra_metrics(results_dir, reference_root=ref_root)

    records: list[dict[str, Any]] = []
    for field in fields:
        rel_l2 = metrics.get(field)
        if rel_l2 is not None:
            records.append({
                "metric": f"spectra_{field}_relative_l2",
                "value": rel_l2,
                "unit": "relative_l2",
            })
            records.append({
                "metric": f"spectra_{field}_score",
                "value": spectra_score(rel_l2),
                "unit": "score_0_1",
            })

    mean_l2 = metrics.get("mean")
    if mean_l2 is not None:
        records.append({
            "metric": "spectra_mean_relative_l2",
            "value": mean_l2,
            "unit": "relative_l2",
        })
        records.append({
            "metric": "spectra_mean_score",
            "value": spectra_score(mean_l2),
            "unit": "score_0_1",
        })

    return records
