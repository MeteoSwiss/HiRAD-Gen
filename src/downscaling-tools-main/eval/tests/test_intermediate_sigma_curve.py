from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


MODULE_PATH = Path(__file__).resolve().parents[1] / "_backends" / "spectra" / "intermediate_sigma_curve.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_intermediate_sigma_curve_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_sigma_curves_reports_full_and_residual_metrics(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "compute_cl_for_field",
        lambda lat, lon, values, nside, lmax: np.asarray(values, dtype=np.float64),
    )

    ds = xr.Dataset(
        data_vars={
            "inter_state": (
                ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
                np.array([[[[[2.0], [2.0]], [[4.0], [4.0]]]]], dtype=np.float32),
            ),
            "x_interp": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[1.0], [1.0]]]], dtype=np.float32),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[4.0], [4.0]]]], dtype=np.float32),
            ),
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.array([[[[5.0], [5.0]]]], dtype=np.float32),
            ),
            "lat_hres": (["grid_point_hres"], np.array([10.0, 11.0], dtype=np.float32)),
            "lon_hres": (["grid_point_hres"], np.array([0.0, 1.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "sampling_step": [0, 1],
            "grid_point_hres": [0, 1],
            "weather_state": ["2t"],
        },
        attrs={"sampling_config_json": '{"sigma_max": 10.0, "sigma_min": 1.0, "rho": 1.0}'},
    )

    result = module.evaluate_sigma_curves(
        ds,
        param="2t",
        target_var="y",
        sample_index=0,
        ensemble_index=0,
        nside=8,
        lmax=8,
    )

    full_records = result["records_by_scope"]["full_field"]
    residual_records = result["records_by_scope"]["residual"]
    assert len(full_records) == 2
    assert len(residual_records) == 2
    assert full_records[0]["relative_l2"] == pytest.approx(0.5)
    assert full_records[1]["relative_l2"] == pytest.approx(0.0)
    assert residual_records[0]["relative_l2"] == pytest.approx(2.0 / 3.0)
    assert residual_records[1]["relative_l2"] == pytest.approx(0.0)


def test_write_scope_outputs_writes_legacy_and_residual_artifacts(tmp_path: Path):
    module = _load_module()
    outputs = module.write_scope_outputs(
        output_dir=tmp_path,
        param="2t",
        target_var="y",
        sigma_schedule=[10.0, 1.0],
        records_by_scope={
            "full_field": [
                {"step": 0, "sigma": 10.0, "mean_abs_log10": 0.1, "rmse": 1.0, "relative_l2": 0.5},
                {"step": 1, "sigma": 1.0, "mean_abs_log10": 0.0, "rmse": 0.0, "relative_l2": 0.0},
            ],
            "residual": [
                {"step": 0, "sigma": 10.0, "mean_abs_log10": 0.2, "rmse": 2.0, "relative_l2": 0.7},
                {"step": 1, "sigma": 1.0, "mean_abs_log10": 0.0, "rmse": 0.0, "relative_l2": 0.0},
            ],
        },
    )

    assert Path(outputs["full_field"]["table"]).name == "spectra_sigma_curve.txt"
    assert Path(outputs["residual"]["table"]).name == "spectra_sigma_curve_residual.txt"
    assert Path(outputs["metadata"]["path"]).exists()
