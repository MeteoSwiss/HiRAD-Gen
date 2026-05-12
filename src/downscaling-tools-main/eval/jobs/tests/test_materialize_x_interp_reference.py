from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "eval/jobs/materialize_x_interp_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("materialize_x_interp_reference", MODULE_PATH)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_prediction(path: Path, *, include_x_interp: bool = True, states: list[str] | None = None) -> None:
    states = states or ["10u", "10v", "2t", "sp", "msl", "t_850", "z_500"]
    shape = (1, 2, 3, len(states))
    y = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    x_interp = y + 10.0
    data_vars = {
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
        "date": (("sample",), np.array([20230826], dtype=np.int64)),
        "lead_step_hours": (("sample",), np.array([24], dtype=np.int64)),
        "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0, 2.0], dtype=np.float32)),
        "lat_hres": (("grid_point_hres",), np.array([45.0, 46.0, 47.0], dtype=np.float32)),
    }
    if include_x_interp:
        data_vars["x_interp"] = (
            ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
            x_interp,
        )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sample": [0],
            "ensemble_member": [1, 2],
            "grid_point_hres": [0, 1, 2],
            "weather_state": states,
        },
        attrs={"run_id": "source_run", "checkpoint_id": "abc"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    ds.close()


def test_materialize_file_sets_y_pred_from_x_interp(tmp_path):
    mod = _load_module()
    src_dir = tmp_path / "src"
    out_root = tmp_path / "out"
    _write_prediction(src_dir / "predictions_20230826_step024.nc")

    summary = mod.materialize_reference(
        src_dir,
        out_root,
        expected_dates=[20230826],
        expected_steps=[24],
        overwrite=True,
    )

    assert summary["files_written"] == 1
    with xr.open_dataset(out_root / "predictions" / "predictions_20230826_step024.nc") as out:
        np.testing.assert_allclose(out["y_pred"].values, out["x_interp"].values)
        assert "y" in out
        assert out.attrs["reference_type"] == "x_interp_o96_to_o320"
        assert out.attrs["scoreboard_candidate_field_source"] == "x_interp"


def test_validate_refuses_missing_x_interp(tmp_path):
    mod = _load_module()
    src_dir = tmp_path / "src"
    _write_prediction(src_dir / "predictions_20230826_step024.nc", include_x_interp=False)

    with pytest.raises(ValueError, match="x_interp"):
        mod.validate_source_predictions(src_dir, expected_dates=[20230826], expected_steps=[24])


def test_validate_refuses_missing_required_weather_state(tmp_path):
    mod = _load_module()
    src_dir = tmp_path / "src"
    _write_prediction(src_dir / "predictions_20230826_step024.nc", states=["2t"])

    with pytest.raises(ValueError, match="required weather states"):
        mod.validate_source_predictions(
            src_dir,
            expected_dates=[20230826],
            expected_steps=[24],
            required_weather_states=["2t", "10u"],
        )
