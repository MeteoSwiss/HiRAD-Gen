from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from . import load_predict_submodule


def test_bundle_key_is_frozen_and_preserves_values(monkeypatch):
    """BundleKey should keep constructor values and reject mutation."""

    mod = load_predict_submodule(monkeypatch, "types")
    key = mod.BundleKey(date="20230826", step=24, member=7)

    assert key.date == "20230826"
    assert key.step == 24
    assert key.member == 7
    with pytest.raises(FrozenInstanceError):
        key.step = 48


def test_bundle_key_hash_matches_for_equal_values(monkeypatch):
    """Equal bundle keys should compare and hash identically."""

    mod = load_predict_submodule(monkeypatch, "types")
    key_a = mod.BundleKey(date="20230826", step=24, member=7)
    key_b = mod.BundleKey(date="20230826", step=24, member=7)

    assert key_a == key_b
    assert hash(key_a) == hash(key_b)
    assert {key_a, key_b} == {key_a}


def test_prediction_config_defaults(monkeypatch):
    """PredictionConfig should expose the documented runtime defaults."""

    mod = load_predict_submodule(monkeypatch, "types")
    config = mod.PredictionConfig(
        checkpoint_path=Path("/checkpoints/last.ckpt"),
        input_dir=Path("/input"),
        output_dir=Path("/output"),
    )

    assert config.device == "cuda"
    assert config.precision == "fp32"
    assert config.num_gpus_per_model == 1
    assert config.validation_frequency == "50h"
    assert config.rank0_write_wait_seconds == 7200
    assert config.members == mod.DEFAULT_MEMBERS
    assert config.steps == mod.DEFAULT_STEPS
    assert config.dates == mod.DEFAULT_DATES
    assert config.members is not mod.DEFAULT_MEMBERS
    assert config.steps is not mod.DEFAULT_STEPS
    assert config.dates is not mod.DEFAULT_DATES


def test_prediction_config_has_no_builtin_value_validation(monkeypatch):
    """PredictionConfig should currently rely on external validation only."""

    mod = load_predict_submodule(monkeypatch, "types")
    config = mod.PredictionConfig(
        checkpoint_path=Path("/checkpoints/last.ckpt"),
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        device="cpu",
        precision="unexpected",
        members=[99],
        steps=[6],
        dates=["20240101"],
    )

    assert config.device == "cpu"
    assert config.precision == "unexpected"
    assert config.members == [99]
    assert config.steps == [6]
    assert config.dates == ["20240101"]
