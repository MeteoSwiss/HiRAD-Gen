from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from . import load_predict_submodule


class _BrokenDataset:
    def to_netcdf(self, path: Path) -> None:
        raise RuntimeError("boom")



def _tiny_dataset() -> xr.Dataset:
    return xr.Dataset({"value": (("sample",), [1, 2, 3])})



def test_rank0_file_writer_creates_output_and_done_marker(monkeypatch, tmp_path: Path):
    """Rank 0 should write the file and emit a done marker."""

    mod = load_predict_submodule(monkeypatch, "distributed_io")
    writer = mod.Rank0FileWriter(global_rank=0, timeout_seconds=1, poll_seconds=0.01)
    out_path = tmp_path / "predictions.nc"

    writer.write_or_wait(_tiny_dataset(), out_path)

    assert out_path.exists()
    assert writer._done_marker(out_path).exists()
    assert not writer._failed_marker(out_path).exists()



def test_rank0_file_writer_waits_for_precreated_done_marker(monkeypatch, tmp_path: Path):
    """Nonzero ranks should return once the done marker already exists."""

    mod = load_predict_submodule(monkeypatch, "distributed_io")
    writer = mod.Rank0FileWriter(global_rank=3, timeout_seconds=1, poll_seconds=0.01)
    out_path = tmp_path / "predictions.nc"
    writer._done_marker(out_path).write_text("ok\n", encoding="utf-8")

    writer.write_or_wait(None, out_path)

    assert writer._done_marker(out_path).exists()



def test_rank0_file_writer_clears_stale_markers(monkeypatch, tmp_path: Path):
    """Marker cleanup should remove stale success and failure markers."""

    mod = load_predict_submodule(monkeypatch, "distributed_io")
    writer = mod.Rank0FileWriter(global_rank=0, timeout_seconds=1, poll_seconds=0.01)
    out_path = tmp_path / "predictions.nc"
    writer._done_marker(out_path).write_text("ok\n", encoding="utf-8")
    writer._failed_marker(out_path).write_text("RuntimeError: stale\n", encoding="utf-8")

    writer._clear_markers(out_path)

    assert not writer._done_marker(out_path).exists()
    assert not writer._failed_marker(out_path).exists()



def test_rank0_file_writer_propagates_failed_marker(monkeypatch, tmp_path: Path):
    """Failure markers written by rank 0 should abort waiting nonzero ranks."""

    mod = load_predict_submodule(monkeypatch, "distributed_io")
    out_path = tmp_path / "predictions.nc"
    rank0_writer = mod.Rank0FileWriter(global_rank=0, timeout_seconds=1, poll_seconds=0.01)

    with pytest.raises(RuntimeError, match="boom"):
        rank0_writer.write_or_wait(_BrokenDataset(), out_path)

    failed_marker = rank0_writer._failed_marker(out_path)
    assert failed_marker.exists()
    assert "RuntimeError: boom" in failed_marker.read_text(encoding="utf-8")

    waiting_writer = mod.Rank0FileWriter(global_rank=1, timeout_seconds=1, poll_seconds=0.01)
    with pytest.raises(SystemExit, match=r"Rank-0 write failed .*RuntimeError: boom"):
        waiting_writer.write_or_wait(None, out_path)
