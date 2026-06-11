"""Tests for TC scoring module."""

from __future__ import annotations

import json

import pytest

from eval._backends.scoreboard.tc import (
    load_tc_extreme_scores_from_json,
    mslp_depth,
    multi_depth_enfo_deviation,
    multi_depth_tc_score,
    normalize_tc_rows,
    rescale_with_eefo_floor,
)


def test_mslp_depth_standard():
    assert mslp_depth(970.0) == pytest.approx(43.25)


def test_mslp_depth_above_reference():
    assert mslp_depth(1020.0) == 0.0


def test_multi_depth_tc_score_perfect():
    """Model matching analysis should score 1.0."""
    analysis = {
        "mslp_p1": 1002.7, "mslp_p01": 993.0, "mslp_min": 971.3,
        "wind_p99": 14.0, "wind_p999": 22.4, "wind_max": 32.0,
    }
    result = multi_depth_tc_score(analysis, analysis)
    assert result == pytest.approx(1.0)


def test_multi_depth_tc_score_partial():
    """Model weaker than analysis should score < 1.0."""
    analysis = {
        "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
        "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
    }
    model = {
        "mslp_p1": 990.0, "mslp_p01": 985.0, "mslp_min": 980.0,
        "wind_p99": 20.0, "wind_p999": 25.0, "wind_max": 30.0,
    }
    result = multi_depth_tc_score(model, analysis)
    assert result is not None
    assert 0.0 < result < 1.0


def test_multi_depth_tc_score_no_data():
    assert multi_depth_tc_score({}, {}) is None


def test_multi_depth_enfo_deviation():
    analysis = {
        "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
        "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
    }
    enfo = {
        "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
        "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
    }
    model = {
        "mslp_p1": 975.0, "mslp_p01": 970.0, "mslp_min": 965.0,
        "wind_p99": 28.0, "wind_p999": 32.0, "wind_max": 36.0,
    }
    result = multi_depth_enfo_deviation(model, enfo, analysis)
    assert result is not None
    assert result > 0.0


def test_rescale_with_eefo_floor():
    assert rescale_with_eefo_floor(0.8, 0.2) == pytest.approx(0.75)
    assert rescale_with_eefo_floor(0.5, None) == pytest.approx(0.5)
    assert rescale_with_eefo_floor(0.1, 0.2) == pytest.approx(0.0)


def test_normalize_tc_rows_analysis_anchored():
    """With explicit canonical analysis, scoring uses multi-depth."""
    analysis = {
        "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
        "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
    }
    rows = [
        dict(exp="OPER_O320_0001", **analysis),
        {
            "exp": "ENFO_O320_0001",
            "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
            "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
        },
        {
            "exp": "manual_model_run",
            "mslp_p1": 975.0, "mslp_p01": 970.0, "mslp_min": 965.0,
            "wind_p99": 28.0, "wind_p999": 32.0, "wind_max": 36.0,
        },
    ]
    normalize_tc_rows(rows, canonical_analysis=analysis)

    oper_row = rows[0]
    assert oper_row["_extreme_score_value"] == 1.0

    model_row = rows[2]
    score = model_row["_extreme_score_value"]
    assert score is not None
    assert 0.5 < score < 1.0


def test_normalize_tc_rows_legacy_fallback():
    """Without tail percentiles, uses batch-relative normalization."""
    rows = [
        {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
        {"exp": "manual_model", "mslp_980_990_fraction": 0.8, "wind_gt_25_fraction": 0.6},
    ]
    normalize_tc_rows(rows)
    assert rows[1]["_extreme_score_value"] == pytest.approx(1.0)


def test_load_tc_extreme_scores_from_json_with_explicit_canonical(tmp_path):
    """Pass canonical analysis explicitly to bypass YAML lookup."""
    stats_path = tmp_path / "tc.stats.json"
    analysis = {
        "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
        "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
    }
    stats_path.write_text(json.dumps({
        "events": {
            "idalia": {
                "extreme_tail": {
                    "rows": [
                        dict(exp="OPER_O320_0001", **analysis),
                        {
                            "exp": "ENFO_O320_0001",
                            "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
                            "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
                        },
                        {
                            "exp": "manual_0c446b41_new_o96_o320",
                            "mslp_p1": 975.0, "mslp_p01": 970.0, "mslp_min": 965.0,
                            "wind_p99": 28.0, "wind_p999": 32.0, "wind_max": 36.0,
                        },
                    ]
                }
            },
        }
    }))

    result = load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_0c446b41_new_o96_o320",
        canonical_analysis_by_event={"idalia": analysis},
    )

    assert 0.5 < result["idalia"] < 1.0
    assert "idalia_enfo_dev" in result
    assert result["idalia_enfo_dev"] > 0.0


def test_load_tc_extreme_scores_custom_events(tmp_path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps({
        "events": {
            "humberto": {
                "extreme_tail": {
                    "rows": [
                        {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                        {"exp": "manual_95a07500_new_o48_o96", "mslp_980_990_fraction": 0.9, "wind_gt_25_fraction": 0.8},
                    ]
                }
            }
        }
    }))

    result = load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_95a07500_new_o48_o96",
        event_names=("humberto",),
        canonical_analysis_by_event={},
    )

    assert result == {"humberto": pytest.approx(1.0)}
