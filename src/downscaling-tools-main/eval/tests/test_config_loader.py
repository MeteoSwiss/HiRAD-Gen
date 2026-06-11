"""Tests for eval.config.loader."""

from __future__ import annotations

import pytest
import yaml

from eval.config.loader import (
    ConfigValidationError,
    load_event,
    load_host,
    load_lane,
)


def test_load_lane_valid():
    config = load_lane("o96_o320")
    assert "predict" in config
    assert "evaluator_groups" in config
    predict = config["predict"]
    assert isinstance(predict["members"], list)
    assert isinstance(predict["steps"], list)
    assert isinstance(predict["dates"], list)


def test_load_lane_with_overrides():
    config = load_lane("o96_o320", overrides={"predict": {"members": [1, 2]}})
    assert config["predict"]["members"] == [1, 2]
    # Original steps should still be present
    assert len(config["predict"]["steps"]) > 0


def test_load_lane_rejects_unknown_key(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {
        "predict": {"members": [1], "steps": [24], "dates": ["20230826"]},
        "evaluator_groups": {"default": ["tc"]},
        "bogus_key": True,
    }
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="unknown top-level key"):
        load_lane("bad")


def test_load_lane_rejects_missing_required(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {"evaluator_groups": {"default": ["tc"]}}
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="missing required top-level key 'predict'"):
        load_lane("bad")


def test_load_lane_rejects_bad_predict_members(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {
        "predict": {"members": ["a", "b"], "steps": [24], "dates": ["20230826"]},
        "evaluator_groups": {"default": ["tc"]},
    }
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="predict.members.*must be a list of int"):
        load_lane("bad")


def test_load_host_valid():
    config = load_host("atos_ac")
    assert "scheduler" in config
    assert "environment_setup" in config
    assert config["scheduler"]["qos"] == "nf"


def test_load_host_rejects_relative_code_root(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "hosts").mkdir()
    bad = {
        "code_root": "relative/path",
        "scratch_root": "/home/user/scratch",
        "scheduler": {"qos": "nf", "default_time": "04:00:00"},
        "environment_setup": {
            "module_loads": ["ecmwf-toolbox"],
            "venv_activate": "/path/to/venv/bin/activate",
        },
    }
    (tmp_path / "hosts" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="code_root.*must be an absolute path"):
        load_host("bad")


def test_load_event_valid():
    config = load_event("idalia")
    assert config["name"] == "idalia"
    assert "lat_min" in config
    assert "lat_max" in config
    assert "lon_min" in config
    assert "lon_max" in config
    assert isinstance(config["dates"], list)


def test_load_lane_allows_resource_profiles():
    config = load_lane("o96_o320")
    assert "resource_profiles" in config
    assert "predict" in config["resource_profiles"]


def test_lane_configs_do_not_advertise_quaver_group():
    for lane in ["o48_o96", "o96_o320", "o320_o1280", "o1280_o2560"]:
        config = load_lane(lane)
        assert "quaver" not in config

        evaluator_groups = config["evaluator_groups"]
        for evaluators in evaluator_groups.values():
            assert "quaver" not in evaluators
