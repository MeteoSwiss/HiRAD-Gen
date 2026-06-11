"""Tests for surface scoring module."""

from __future__ import annotations

import json

import pytest

from eval._backends.scoreboard.surface import (
    format_surface_loss_for_scoreboard,
    load_surface_loss_metrics,
    surface_weighted_nmse,
)


def test_surface_weighted_nmse_with_truth_std():
    variables = {
        "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
        "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
        "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
    }
    truth_std = {"msl": 10.0, "sp": 5.0, "2t": 2.0}

    result, per_var = surface_weighted_nmse(variables, truth_std_by_variable=truth_std)

    assert result is not None
    assert result == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert per_var["msl"] == pytest.approx(0.1)
    assert per_var["sp"] == pytest.approx(0.2)
    assert per_var["2t"] == pytest.approx(0.5)


def test_surface_weighted_nmse_with_existing_nmse():
    variables = {
        "msl": {"mean_nmse": 0.1, "normalized_weight": 0.6},
        "sp": {"mean_nmse": 0.2, "normalized_weight": 0.4},
    }
    result, per_var = surface_weighted_nmse(variables)
    assert result == pytest.approx(0.6 * 0.1 + 0.4 * 0.2)


def test_surface_weighted_nmse_empty():
    result, per_var = surface_weighted_nmse({})
    assert result is None
    assert per_var == {}


def test_load_surface_loss_metrics(tmp_path):
    summary = tmp_path / "surface_loss_summary.json"
    summary.write_text(json.dumps({
        "weighted_surface_mse": 10.0,
        "variables": {
            "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
            "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
            "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
        },
    }))

    result = load_surface_loss_metrics(
        summary,
        truth_std_by_variable={"msl": 10.0, "sp": 5.0, "2t": 2.0},
    )

    assert result["weighted_mse"] == pytest.approx(10.0)
    assert result["weighted_nmse"] == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert [e["variable"] for e in result["top_contributors"]] == ["msl", "sp"]


def test_format_surface_loss_for_scoreboard():
    assert format_surface_loss_for_scoreboard({"weighted_nmse": 0.17}) == "0.1700"
    assert format_surface_loss_for_scoreboard({"weighted_mse": 1.23e-5}) == "1.230000e-05"
    assert format_surface_loss_for_scoreboard({}) == "na"
