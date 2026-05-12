"""Tests for eval.scoreboard.formatter."""

from __future__ import annotations

import csv

from eval.scoreboard.formatter import COLUMNS, to_csv, to_markdown
from eval.scoreboard.types import ScoreRecord


def _sample_scores():
    return [
        ScoreRecord(evaluator="tc", metric="idalia_extreme", value=0.978, unit="score"),
        ScoreRecord(evaluator="spectra", metric="mean_l2", value=0.032, unit="distance"),
    ]


def test_to_csv_writes_headers_and_rows(tmp_path):
    out = tmp_path / "scores.csv"
    to_csv(_sample_scores(), out)

    with open(out) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == COLUMNS
    assert len(rows) == 2
    assert rows[0][0] == "tc"
    assert rows[0][1] == "idalia_extreme"
    assert rows[1][0] == "spectra"


def test_to_csv_empty_scores_writes_headers(tmp_path):
    out = tmp_path / "empty.csv"
    to_csv([], out)

    with open(out) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == COLUMNS
    assert rows == []


def test_to_markdown_writes_table(tmp_path):
    out = tmp_path / "scores.md"
    to_markdown(_sample_scores(), out)

    text = out.read_text()
    lines = text.strip().split("\n")
    assert len(lines) >= 3  # header + separator + at least 1 data row
    assert "evaluator" in lines[0]
    assert "metric" in lines[0]
    assert "---" in lines[1]
    assert "tc" in lines[2]
    assert "0.978000" in lines[2]
