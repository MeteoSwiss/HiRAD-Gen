from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from . import load_predict_submodule


def test_parse_bundle_key_valid_filename(monkeypatch):
    """parse_bundle_key should extract date, step, and member from valid names."""

    mod = load_predict_submodule(monkeypatch, "bundle_manager")
    path = Path("/bundles/prefix_date20230826_time1200_mem7_step024h_input_bundle.nc")

    assert mod.parse_bundle_key(path) == mod.BundleKey(date="20230826", step=24, member=7)


@pytest.mark.parametrize(
    "filename",
    [
        "date20230826_time1200_mem7_step24h_input_bundle.nc",
        "date20230826_time1200_mem7_step024_input_bundle.nc",
        "date20230826_mem7_step024h_input_bundle.nc",
        "predictions_20230826_step024.nc",
    ],
)
def test_parse_bundle_key_invalid_filenames_return_none(monkeypatch, filename):
    """parse_bundle_key should return None when the filename pattern does not match."""

    mod = load_predict_submodule(monkeypatch, "bundle_manager")

    assert mod.parse_bundle_key(Path(filename)) is None


def test_parse_bundle_key_handles_weird_paths_and_non_matches(monkeypatch):
    """parse_bundle_key should only inspect the basename and ignore non-matching parents."""

    mod = load_predict_submodule(monkeypatch, "bundle_manager")

    valid = Path("/odd/date20200101_time0000_mem1_step024h_input_bundle.nc")
    invalid = Path("/odd/date20200101_time0000_mem1_step024h_input_bundle.nc/not-a-bundle.txt")

    assert mod.parse_bundle_key(valid) == mod.BundleKey(date="20200101", step=24, member=1)
    assert mod.parse_bundle_key(invalid) is None


@pytest.mark.parametrize(
    ("date_str", "expected"),
    [
        ("20230826", np.datetime64("2023-08-26T00:00:00", "ns")),
        ("20240229", np.datetime64("2024-02-29T00:00:00", "ns")),
        ("19991231", np.datetime64("1999-12-31T00:00:00", "ns")),
    ],
)
def test_date_str_to_datetime64_converts_expected_values(monkeypatch, date_str, expected):
    """date_str_to_datetime64 should return nanosecond datetime64 values."""

    mod = load_predict_submodule(monkeypatch, "bundle_manager")

    result = mod.date_str_to_datetime64(date_str)

    assert result == expected
    assert np.asarray(result).dtype == np.dtype("datetime64[ns]")


def test_discover_bundles_recursively_prefers_newest_duplicate(monkeypatch, tmp_path: Path):
    """discover_bundles should recurse and keep the newest path for duplicate keys."""

    mod = load_predict_submodule(monkeypatch, "bundle_manager")

    older = tmp_path / "old" / "date20230826_time0000_mem1_step024h_input_bundle.nc"
    newer = tmp_path / "new" / "date20230826_time0000_mem1_step024h_input_bundle.nc"
    other = tmp_path / "nested" / "date20230827_time0600_mem2_step048h_input_bundle.nc"
    ignored = tmp_path / "nested" / "not_a_bundle.nc"

    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)

    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    other.write_text("other", encoding="utf-8")
    ignored.write_text("ignored", encoding="utf-8")

    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    os.utime(other, (150, 150))

    bundle_map = mod.discover_bundles(tmp_path)

    assert bundle_map[mod.BundleKey("20230826", 24, 1)] == newer
    assert bundle_map[mod.BundleKey("20230827", 48, 2)] == other
    assert len(bundle_map) == 2
