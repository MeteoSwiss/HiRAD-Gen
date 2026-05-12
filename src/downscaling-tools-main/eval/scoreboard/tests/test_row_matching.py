"""Tests for row classification and matching."""

from __future__ import annotations

from eval._backends.scoreboard.row_matching import (
    RowClassification,
    classify_row,
    extract_checkpoint_token,
    find_model_row,
    tc_candidates,
)
from eval.scoreboard.types import RowClassification as RowClassEnum


def test_classify_analysis():
    assert classify_row("OPER_O320_0001") == RowClassification.ANALYSIS
    assert classify_row("oper_o1280") == RowClassification.ANALYSIS


def test_classify_reference():
    assert classify_row("ENFO_O320_0001") == RowClassification.REFERENCE
    assert classify_row("IP6Y_run") == RowClassification.REFERENCE


def test_classify_eefo():
    assert classify_row("EEFO_O96") == RowClassification.EEFO


def test_classify_model():
    assert classify_row("manual_0c446b41_new_o96_o320") == RowClassification.MODEL
    assert classify_row("some_experiment") == RowClassification.MODEL


def test_classify_custom_prefix_map():
    custom = {"CUSTOM": RowClassification.ANALYSIS}
    assert classify_row("CUSTOM_run", prefix_map=custom) == RowClassification.ANALYSIS
    assert classify_row("OPER_run", prefix_map=custom) == RowClassification.MODEL


def test_extract_checkpoint_token():
    assert extract_checkpoint_token("manual_0c446b41_new_o96_o320") == "0c446b41"
    assert extract_checkpoint_token("39991df81216460fb7f3bd048df733c3") == "39991df81216460fb7f3bd048df733c3"
    assert extract_checkpoint_token("no_hex_here") is None


def test_tc_candidates():
    candidates = tc_candidates("manual_0c446b41_new_o96_o320")
    assert "manual_0c446b41_new_o96_o320" in candidates
    assert "0c446b41" in candidates


def test_find_model_row_exact_match():
    rows = [
        {"exp": "OPER_O320_0001"},
        {"exp": "ENFO_O320_0001"},
        {"exp": "manual_abc12345_new"},
    ]
    result = find_model_row(rows, "manual_abc12345_new")
    assert result is not None
    assert result["exp"] == "manual_abc12345_new"


def test_find_model_row_substring_match():
    rows = [
        {"exp": "ENFO_O320"},
        {"exp": "manual_0c446b41_new_o96_o320_20260317_piecewise30"},
    ]
    result = find_model_row(rows, "manual_0c446b41_new_o96_o320")
    assert result is not None
    assert "0c446b41" in result["exp"]


def test_find_model_row_sole_non_reference():
    rows = [
        {"exp": "ENFO_O320"},
        {"exp": "some_model_run"},
    ]
    result = find_model_row(rows, "totally_different_id")
    assert result is not None
    assert result["exp"] == "some_model_run"


def test_find_model_row_no_match():
    rows = [
        {"exp": "ENFO_O320"},
        {"exp": "model_a"},
        {"exp": "model_b"},
    ]
    result = find_model_row(rows, "totally_different_id")
    assert result is None
