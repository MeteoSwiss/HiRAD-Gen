"""Row classification and matching for TC stats rows."""

from __future__ import annotations

import re
from typing import Any

from eval.scoreboard.types import RowClassification

# Default prefix→classification mapping. Callers can override.
DEFAULT_PREFIX_MAP: dict[str, RowClassification] = {
    "OPER": RowClassification.ANALYSIS,
    "ENFO": RowClassification.REFERENCE,
    "IP6Y": RowClassification.REFERENCE,
    "EEFO": RowClassification.EEFO,
}

CHECKPOINT_TOKEN_RE = re.compile(r"(?:^|manual_)([0-9a-f]{7,64})(?:_|$)")


def classify_row(
    exp_label: str,
    *,
    prefix_map: dict[str, RowClassification] | None = None,
) -> RowClassification:
    """Classify a TC stats row by its experiment label prefix."""
    mapping = prefix_map or DEFAULT_PREFIX_MAP
    upper = exp_label.strip().upper()
    for prefix, classification in mapping.items():
        if upper.startswith(prefix.upper()):
            return classification
    return RowClassification.MODEL


def is_analysis_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.ANALYSIS


def is_reference_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.REFERENCE


def is_eefo_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.EEFO


def find_row_by_predicate(
    rows: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    for row in rows:
        if predicate(str(row.get("exp", "")).strip()):
            return row
    return None


def extract_checkpoint_token(text: str) -> str | None:
    match = CHECKPOINT_TOKEN_RE.search(str(text).strip())
    if match:
        return match.group(1)
    return None


def tc_candidates(run_id: str) -> list[str]:
    """Generate candidate labels for matching a run_id against TC rows."""
    candidates = [run_id.strip()]
    token = extract_checkpoint_token(run_id)
    if token:
        candidates.extend({token, token[:8], token[:7]})
    return [c for c in candidates if c]


def find_model_row(
    rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any] | None:
    """Find the TC stats row matching a given run_id.

    Tries exact match, then substring match, then falls back to
    the sole non-reference row if only one exists.
    """
    candidates = tc_candidates(run_id)

    for candidate in candidates:
        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if exp == candidate:
                return row

    for candidate in candidates:
        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if candidate in exp or exp in candidate:
                return row

    non_reference = [
        row for row in rows
        if classify_row(str(row.get("exp", "")).strip()) == RowClassification.MODEL
    ]
    if len(non_reference) == 1:
        return non_reference[0]

    return None
