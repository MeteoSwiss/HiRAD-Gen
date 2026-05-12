"""Tests for eval.scoreboard._utils — finite_float and load_json."""
from __future__ import annotations

import json

from eval._backends.scoreboard._utils import finite_float, load_json


def test_finite_float_accepts_finite_int():
    assert finite_float(3) == 3.0


def test_finite_float_accepts_finite_str():
    assert finite_float("1.5") == 1.5


def test_finite_float_rejects_nan():
    assert finite_float(float("nan")) is None


def test_finite_float_rejects_inf():
    assert finite_float(float("inf")) is None


def test_finite_float_rejects_non_numeric():
    assert finite_float("abc") is None
    assert finite_float(None) is None


def test_load_json_returns_dict(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1}))
    assert load_json(p) == {"a": 1}


def test_load_json_returns_empty_for_non_dict(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert load_json(p) == {}
