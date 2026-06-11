"""Tests for eval.discovery.predictions."""

from __future__ import annotations

from eval.discovery.predictions import PREDICTION_RE, PredictionFile, find_predictions


def test_find_predictions_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_predictions(empty) == []


def test_find_predictions_canonical_names(predictions_dir):
    result = find_predictions(predictions_dir)
    assert len(result) == 6  # 2 dates x 3 steps
    assert all(isinstance(p, PredictionFile) for p in result)
    assert result[0].date == "20230826"
    assert result[0].step == 24
    assert result[0].member == 0


def test_find_predictions_ignores_non_matching(predictions_dir):
    result = find_predictions(predictions_dir)
    names = {p.path.name for p in result}
    assert "random_file.txt" not in names
    assert "predictions_bad_name.nc" not in names
    for p in result:
        assert PREDICTION_RE.match(p.path.name)


def test_find_predictions_sorts_by_date_and_step(predictions_dir):
    result = find_predictions(predictions_dir)
    keys = [(p.date, p.step) for p in result]
    assert keys == sorted(keys)
