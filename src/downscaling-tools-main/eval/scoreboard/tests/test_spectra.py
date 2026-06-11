"""Tests for spectra scoring module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval._backends.scoreboard.spectra import (
    finite_positive_mask,
    load_spectra_metrics,
    relative_l2,
    relative_l2_weighted,
    spectra_score,
)


def test_spectra_score():
    assert spectra_score(0.0) == pytest.approx(1.0)
    assert spectra_score(0.5) == pytest.approx(0.5)
    assert spectra_score(1.0) == pytest.approx(0.0)
    assert spectra_score(1.5) == pytest.approx(0.0)  # clamped


def test_relative_l2_identical():
    arr = np.ones(200, dtype=np.float64)
    assert relative_l2(arr, arr) == pytest.approx(0.0)


def test_relative_l2_weighted_identical():
    arr = np.ones(200, dtype=np.float64)
    assert relative_l2_weighted(arr, arr) == pytest.approx(0.0)


def test_relative_l2_weighted_with_wavenumbers():
    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    pred = np.ones(200, dtype=np.float64)
    truth = np.ones(200, dtype=np.float64)
    pred[149] = 2.0  # wavenumber 150 — should be scored
    result = relative_l2_weighted(pred, truth, wavenumbers=wvn)
    assert result > 0.0


def test_relative_l2_weighted_ignores_low_wavenumbers():
    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    pred = np.ones(200, dtype=np.float64)
    truth = np.ones(200, dtype=np.float64)
    pred[49] = 5.0  # wavenumber 50 — below threshold, should be ignored
    result = relative_l2_weighted(pred, truth, wavenumbers=wvn)
    assert result == pytest.approx(0.0)


def test_finite_positive_mask_adaptive_threshold_low_lmax():
    """When max wavenumber <= 100, threshold drops to max_wvn/3."""
    wvn = np.arange(0, 96, dtype=np.float64)
    arr = np.ones_like(wvn) * 10.0
    mask = finite_positive_mask(arr, wavenumbers=wvn)
    assert mask.sum() > 0
    assert mask[0] is np.False_
    assert mask[32] is np.True_


def test_finite_positive_mask_standard_threshold():
    wvn = np.arange(0, 321, dtype=np.float64)
    arr = np.ones_like(wvn) * 10.0
    mask = finite_positive_mask(arr, wavenumbers=wvn)
    assert mask[100] is np.False_
    assert mask[101] is np.True_


def test_load_spectra_metrics_from_raw_arrays(tmp_path):
    spectra_dir = tmp_path / "spectra"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(json.dumps({
        "dates": [20230826],
        "steps_hours": [120],
        "ensemble_members": [1],
        "template_root": str(ref_root),
    }))

    from eval._backends.scoreboard.spectra import RAW_FIELD_DIRS

    for field_dir in RAW_FIELD_DIRS.values():
        run_dir = spectra_dir / field_dir
        ref_dir = ref_root / field_dir
        run_dir.mkdir()
        ref_dir.mkdir()
        curve = np.ones(200, dtype=np.float64)
        np.save(run_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)
        np.save(ref_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)

    result = load_spectra_metrics(spectra_dir)

    assert result["10u"] is not None
    assert result["mean"] is not None
    assert result["mean"] == pytest.approx(0.0)


def test_load_spectra_metrics_empty_dir(tmp_path):
    spectra_dir = tmp_path / "empty_spectra"
    spectra_dir.mkdir()
    result = load_spectra_metrics(spectra_dir)
    assert result["mean"] is None
    assert result["coverage"] == "missing"
