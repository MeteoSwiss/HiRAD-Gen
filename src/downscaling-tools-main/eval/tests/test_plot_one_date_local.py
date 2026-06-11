from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from eval._backends.region_plotting.plot_one_date_local import DEFAULT_LOCAL_PLOT_OUT_SUBDIR, render_one_date_local_plots


def _write_prediction_file(path: Path, lead_h: int) -> None:
    lon_lres = np.array([-75.0, -70.0, -65.0, -60.0], dtype=np.float32)
    lat_lres = np.array([-10.0, -2.0, 2.0, 10.0], dtype=np.float32)
    lon_hres = np.array([-75.0, -72.5, -70.0, -67.5, -65.0, -62.5], dtype=np.float32)
    lat_hres = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, -2.0], dtype=np.float32)
    weather_states = ["10u", "10v", "2t"]

    x = np.zeros((1, 4, 3), dtype=np.float32)
    y = np.zeros((1, 1, 6, 3), dtype=np.float32)
    y_pred = np.zeros((1, 1, 6, 3), dtype=np.float32)
    for w_idx in range(len(weather_states)):
        x[0, :, w_idx] = np.linspace(0.0 + w_idx, 3.0 + w_idx, 4, dtype=np.float32)
        y[0, 0, :, w_idx] = np.linspace(0.0 + w_idx, 5.0 + w_idx, 6, dtype=np.float32)
        y_pred[0, 0, :, w_idx] = y[0, 0, :, w_idx] + 0.25

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), x, {"lon": "lon_lres", "lat": "lat_lres"}),
            "y": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                y,
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                y_pred,
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_lres": (("grid_point_lres",), lon_lres),
            "lat_lres": (("grid_point_lres",), lat_lres),
            "lon_hres": (("grid_point_hres",), lon_hres),
            "lat_hres": (("grid_point_hres",), lat_hres),
            "date": (("sample",), np.array(["2023-08-26T00:00:00"], dtype="datetime64[ns]")),
            "init_date": (("sample",), np.array(["2023-08-21T00:00:00"], dtype="datetime64[ns]")),
            "lead_step_hours": (("sample",), np.array([lead_h], dtype=np.int32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [1],
            "grid_point_lres": np.arange(4),
            "grid_point_hres": np.arange(6),
            "weather_state": weather_states,
        },
        attrs={"grid": "O320"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_render_one_date_local_plots_writes_baseline_outputs(tmp_path: Path):
    run_root = tmp_path / "run"
    predictions_dir = run_root / "predictions"
    _write_prediction_file(predictions_dir / "predictions_20230826_step024.nc", lead_h=24)
    _write_prediction_file(predictions_dir / "predictions_20230826_step048.nc", lead_h=48)

    outputs = render_one_date_local_plots(
        run_root=run_root,
        date="20230826",
        expected_count=2,
        run_label="toy_run",
    )

    assert len(outputs) == 2
    pdfs = [pdf for pdf, _ in outputs]
    pngs = [png for _, png in outputs]
    assert pdfs[0].name == "amazon_forest_member01_baseline.pdf"
    assert pngs[0] is not None and pngs[0].name == "amazon_forest_member01_baseline.png"
    for pdf, png in outputs:
        assert pdf.exists()
        assert png is not None and png.exists()
        assert pdf.relative_to(run_root).parts[0] == DEFAULT_LOCAL_PLOT_OUT_SUBDIR
        assert png.relative_to(run_root).parts[0] == DEFAULT_LOCAL_PLOT_OUT_SUBDIR
