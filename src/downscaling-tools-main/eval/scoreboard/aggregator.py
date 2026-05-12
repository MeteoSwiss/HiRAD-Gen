"""Scoreboard aggregator — collect scores from evaluator modules."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

from eval.scoreboard.types import ScoreRecord

LOG = logging.getLogger(__name__)

KNOWN_EVALUATORS = ["tc", "spectra", "surface", "sigma", "region_plot"]


def aggregate_scores(
    eval_dir: Path,
    lane_config: dict,
    evaluators: list[str] | None = None,
) -> list[ScoreRecord]:
    """Collect scores from evaluator score() functions.

    Scans eval_dir/evaluators/<name>/ for each known evaluator,
    calls score() on those with scoreboard=True in their EVALUATOR_SPEC,
    and returns sorted ScoreRecord list.

    Args:
        eval_dir: Root evaluation directory containing evaluator outputs.
        lane_config: Lane configuration dict.
        evaluators: Optional filter — only include these evaluator names.
                    None means all scoreboard-eligible evaluators with results.

    Returns:
        List of ScoreRecord sorted by (evaluator, metric).
    """
    eval_dir = Path(eval_dir)
    target_evaluators = evaluators if evaluators is not None else KNOWN_EVALUATORS

    all_records: list[ScoreRecord] = []

    for name in target_evaluators:
        if name not in KNOWN_EVALUATORS:
            LOG.warning("Unknown evaluator: %s (not in KNOWN_EVALUATORS)", name)
            continue

        # Import evaluator module
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError:
            LOG.warning("Cannot import evaluator module: eval.evaluators.%s", name)
            continue

        # Check scoreboard eligibility
        spec = getattr(mod, "EVALUATOR_SPEC", {})
        if not spec.get("scoreboard", False):
            continue

        # Check results directory exists
        results_dir = eval_dir / "evaluators" / name
        if not results_dir.is_dir():
            continue

        # Call score()
        score_fn = getattr(mod, "score", None)
        if score_fn is None:
            LOG.warning("Evaluator %s has no score() function", name)
            continue

        eval_config = lane_config.get(name, {})
        try:
            raw_scores = score_fn(results_dir, lane_config, eval_config)
        except Exception:
            LOG.warning("Evaluator %s score() failed", name, exc_info=True)
            continue

        # Convert dicts to ScoreRecords
        for record in raw_scores:
            all_records.append(ScoreRecord(
                evaluator=name,
                metric=record["metric"],
                value=record["value"],
                unit=record["unit"],
            ))

    all_records.sort(key=lambda r: (r.evaluator, r.metric))
    return all_records
