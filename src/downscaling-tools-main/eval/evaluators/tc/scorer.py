"""TC evaluator — pure scoring math.

Wraps eval.scoreboard.tc scoring functions with the standard evaluator
interface: returns list[dict] of {"metric", "value", "unit"} records.

Scoring algorithms are identical to scoreboard/tc.py — this module
imports them directly to guarantee mathematical equivalence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard.tc import (
    load_tc_extreme_scores_from_json,
    multi_depth_tc_score,
    normalize_tc_rows,
)

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    run_id: str = "",
    stats_filename: str = "stats.json",
    **kwargs,
) -> list[dict[str, Any]]:
    """Score TC results from a stats JSON.

    Parameters
    ----------
    results_dir : Path to the evaluator output directory containing stats JSON.
    lane_config : Full lane configuration dict.
    eval_config : TC-specific config (lane_config["tc"]).
    run_id : Experiment run ID for row matching.
    stats_filename : Name of the stats JSON file.

    Returns
    -------
    List of {"metric": str, "value": float, "unit": str} records.
    """
    results_dir = Path(results_dir)
    stats_path = results_dir / stats_filename

    if not stats_path.exists():
        LOG.warning("TC stats file not found: %s", stats_path)
        return []

    event_names = eval_config.get("events", [])
    if not event_names:
        LOG.warning("No TC events configured for scoring")
        return []

    scores = load_tc_extreme_scores_from_json(
        stats_path,
        run_id=run_id,
        event_names=event_names,
    )

    records: list[dict[str, Any]] = []
    for event_name in event_names:
        if event_name in scores:
            records.append({
                "metric": f"tc_{event_name}_extreme_score",
                "value": scores[event_name],
                "unit": "score_0_1",
            })
        enfo_dev_key = f"{event_name}_enfo_dev"
        if enfo_dev_key in scores:
            records.append({
                "metric": f"tc_{event_name}_enfo_deviation",
                "value": scores[enfo_dev_key],
                "unit": "deviation",
            })

    # Compute aggregate TC score (mean of per-event extreme scores)
    event_scores = [r["value"] for r in records if r["metric"].endswith("_extreme_score")]
    if event_scores:
        records.append({
            "metric": "tc_mean_extreme_score",
            "value": sum(event_scores) / len(event_scores),
            "unit": "score_0_1",
        })

    return records
