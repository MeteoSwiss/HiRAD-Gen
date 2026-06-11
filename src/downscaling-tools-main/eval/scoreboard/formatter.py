"""Scoreboard formatter — CSV, markdown, and text output."""
from __future__ import annotations

import csv
from pathlib import Path

from eval.scoreboard.types import ScoreRecord

COLUMNS = ["evaluator", "metric", "value", "unit"]


def to_csv(scores: list[ScoreRecord], output_path: Path) -> Path:
    """Write scores as CSV with full float precision.

    Writes headers even if scores is empty.
    Returns output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for s in scores:
            writer.writerow([s.evaluator, s.metric, repr(s.value), s.unit])
    return output_path


def to_markdown(scores: list[ScoreRecord], output_path: Path) -> Path:
    """Write scores as aligned markdown table with 6-decimal precision.

    Returns output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not scores:
        output_path.write_text(
            "| evaluator | metric | value | unit |\n"
            "|-----------|--------|------:|------|\n"
        )
        return output_path

    # Compute column widths
    eval_w = max(len("evaluator"), max(len(s.evaluator) for s in scores))
    metric_w = max(len("metric"), max(len(s.metric) for s in scores))
    unit_w = max(len("unit"), max(len(s.unit) for s in scores))
    value_w = max(len("value"), 13)  # 6 decimals + sign + digits

    lines = []
    lines.append(
        f"| {'evaluator':<{eval_w}} "
        f"| {'metric':<{metric_w}} "
        f"| {'value':>{value_w}} "
        f"| {'unit':<{unit_w}} |"
    )
    lines.append(
        f"|{'-' * (eval_w + 2)}"
        f"|{'-' * (metric_w + 2)}"
        f"|{'-' * (value_w + 1)}:"
        f"|{'-' * (unit_w + 2)}|"
    )
    for s in scores:
        lines.append(
            f"| {s.evaluator:<{eval_w}} "
            f"| {s.metric:<{metric_w}} "
            f"| {s.value:>{value_w}.6f} "
            f"| {s.unit:<{unit_w}} |"
        )

    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def to_pretty_text(scores: list[ScoreRecord]) -> str:
    """Format scores as aligned text for terminal display.

    Returns formatted string.
    """
    if not scores:
        return "No scores."

    eval_w = max(len(s.evaluator) for s in scores)
    metric_w = max(len(s.metric) for s in scores)
    unit_w = max(len(s.unit) for s in scores)

    lines = []
    for s in scores:
        lines.append(
            f"  {s.evaluator:<{eval_w}}  {s.metric:<{metric_w}}  "
            f"{s.value:>13.6f}  {s.unit:<{unit_w}}"
        )
    return "\n".join(lines)
