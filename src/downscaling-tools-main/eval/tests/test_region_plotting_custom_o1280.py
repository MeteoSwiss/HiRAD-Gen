from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval._backends.region_plotting.local_plotting import (
    ensure_member_zero_plot_variables,
    ensure_x_interp_for_plotting,
    get_plot_data_array,
    plot_variable_title,
)
from eval._backends.region_plotting import plot_regions as mod


def test_sample_meta_title_includes_indices_and_dates():
    ds = xr.Dataset(
        data_vars={
            "date": ("sample", np.array(["2026-02-27T00:00:00"], dtype="datetime64[ns]")),
            "init_date": ("sample", np.array(["2026-02-26T00:00:00"], dtype="datetime64[ns]")),
            "lead_step_hours": ("sample", np.array([24], dtype=np.int32)),
        },
        coords={"sample": [0]},
    )
    title = mod._sample_meta_title(ds, "tibet_karakoram", 0)
    assert "tibet_karakoram" in title
    assert "sample_pos=0" in title
    assert "sample_id=0" in title
    assert "date=2026-02-27 00:00" in title
    assert "init=2026-02-26 00:00" in title
    assert "lead_h=24" in title


def test_run_region_plots_uses_custom_path_for_o1280(tmp_path: Path, monkeypatch):
    run_parent = tmp_path / "eval"
    run_name = "myrun"
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True)
    pred_path = run_dir / "predictions.nc"

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y": (("sample", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y_pred": (("sample", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "grid_point_hres": [0],
            "grid_point_lres": [0],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O1280"},
    )
    ds.to_netcdf(pred_path)
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("base\n", encoding="utf-8")

    called = {"custom": False, "default": False}

    def _fake_custom(**kwargs):
        called["custom"] = True

    class _FakeLIP:
        def __init__(self, *args, **kwargs):
            called["default"] = True
            self.regions = []

        def save_plot(self, *args, **kwargs):
            called["default"] = True

    monkeypatch.setattr(mod, "_save_custom_o1280_plots", _fake_custom)
    monkeypatch.setattr(mod, "LocalInferencePlotter", _FakeLIP)

    mod.run_region_plots_from_predictions(
        run_parent_dir=run_parent,
        run_name=run_name,
        predictions_filename="predictions.nc",
    )

    assert called["custom"] is True
    assert called["default"] is False


def test_run_region_plots_accepts_slim_prediction_variables(tmp_path: Path, monkeypatch):
    run_parent = tmp_path / "eval"
    run_name = "slimrun"
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True)
    pred_path = run_dir / "predictions.nc"

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_hres": [0],
            "grid_point_lres": [0],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O320"},
    )
    ds.to_netcdf(pred_path)

    captured: dict[str, list[str]] = {}

    class _FakeLIP:
        def __init__(self, *args, **kwargs):
            self.regions = ["amazon_forest"]

        def save_plot(self, list_regions, list_model_variables, weather_states, **kwargs):
            captured["regions"] = list_regions
            captured["model_variables"] = list_model_variables
            captured["weather_states"] = weather_states

    monkeypatch.setattr(mod, "LocalInferencePlotter", _FakeLIP)

    mod.run_region_plots_from_predictions(
        run_parent_dir=run_parent,
        run_name=run_name,
        predictions_filename="predictions.nc",
    )

    assert captured["regions"] == ["amazon_forest"]
    assert captured["model_variables"] == ["x_0", "y_0", "y_pred_0"]
    assert captured["weather_states"] == ["10u"]


def test_run_region_plots_prefers_x_interp_when_available(tmp_path: Path, monkeypatch):
    run_parent = tmp_path / "eval"
    run_name = "interp_run"
    run_dir = run_parent / run_name
    run_dir.mkdir(parents=True)
    pred_path = run_dir / "predictions.nc"

    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 1, 1), dtype=np.float32)),
            "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 1, 1, 1), dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_hres": [0],
            "grid_point_lres": [0],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O320"},
    )
    ds.to_netcdf(pred_path)

    captured: dict[str, list[str]] = {}

    class _FakeLIP:
        def __init__(self, *args, **kwargs):
            self.regions = ["amazon_forest"]

        def save_plot(self, list_regions, list_model_variables, weather_states, **kwargs):
            captured["regions"] = list_regions
            captured["model_variables"] = list_model_variables
            captured["weather_states"] = weather_states

    monkeypatch.setattr(mod, "LocalInferencePlotter", _FakeLIP)

    mod.run_region_plots_from_predictions(
        run_parent_dir=run_parent,
        run_name=run_name,
        predictions_filename="predictions.nc",
    )

    assert captured["regions"] == ["amazon_forest"]
    assert captured["model_variables"] == ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"]
    assert captured["weather_states"] == ["10u"]


def test_select_prediction_variables_accepts_derived_residuals(tmp_path: Path):
    pred_path = tmp_path / "predictions.nc"
    ds = xr.Dataset(
        data_vars={
            "x": (
                ("sample", "grid_point_lres", "weather_state"),
                np.ones((1, 2, 1), dtype=np.float32),
                {"lon": "lon_lres", "lat": "lat_lres"},
            ),
            "x_interp": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.ones((1, 2, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 2, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.full((1, 2, 2, 1), 0.5, dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0, 1],
            "grid_point_lres": [0, 1],
            "grid_point_hres": [0, 1],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O1280"},
    )
    ds.to_netcdf(pred_path)

    with xr.open_dataset(pred_path) as ds_loaded:
        ds_member = ensure_member_zero_plot_variables(ds_loaded.isel(sample=0, ensemble_member=0))
        selected_variables, selected_weather_states = mod._select_prediction_variables(
            ds_member,
            model_variables=[
                "x_0",
                "x_interp_0",
                "y_0",
                "y_pred_0",
                "residuals_0",
                "residuals_pred_0",
            ],
        )

    assert selected_variables == [
        "x_0",
        "x_interp_0",
        "y_0",
        "y_pred_0",
        "residuals_0",
        "residuals_pred_0",
    ]
    assert selected_weather_states == ["10u"]


def test_region_boxes_accept_o48_helper_regions():
    boxes = mod._region_boxes_for_names(
        [
            "amazon_forest_core",
            "eastern_us_coast",
            "andes_central",
            "himalayas_central",
            "maritime_continent",
            "congo_basin",
        ],
        grid="O96",
    )
    assert sorted(boxes) == [
        "amazon_forest_core",
        "andes_central",
        "congo_basin",
        "eastern_us_coast",
        "himalayas_central",
        "maritime_continent",
    ]


def test_get_plot_data_array_builds_residual_panels():
    ds = xr.Dataset(
        data_vars={
            "x_interp": (
                ("grid_point_hres", "weather_state"),
                np.array([[2.0], [4.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y": (
                ("grid_point_hres", "weather_state"),
                np.array([[1.0], [2.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("grid_point_hres", "weather_state"),
                np.array([[1.5], [3.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
        },
        coords={"grid_point_hres": [0, 1], "weather_state": ["10u"]},
    )

    pred_residual = get_plot_data_array(ds, "x_interp_minus_y_pred")
    truth_residual = get_plot_data_array(ds, "x_interp_minus_y")
    pred_residual_alias = get_plot_data_array(ds, "residuals_pred")
    truth_residual_alias = get_plot_data_array(ds, "residuals")

    np.testing.assert_allclose(pred_residual.values[:, 0], np.array([0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(truth_residual.values[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(pred_residual_alias.values[:, 0], np.array([0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(truth_residual_alias.values[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    assert plot_variable_title("x_interp_minus_y_pred") == "residuals_pred"
    assert plot_variable_title("x_interp_minus_y") == "residuals"
    assert plot_variable_title("residuals_pred") == "residuals_pred"
    assert plot_variable_title("residuals") == "residuals"


def test_member_zero_aliases_and_residual_aliases_are_available():
    ds = xr.Dataset(
        data_vars={
            "x": (
                ("grid_point_lres", "weather_state"),
                np.array([[1.0], [2.0]], dtype=np.float32),
                {"lon": "lon_lres", "lat": "lat_lres"},
            ),
            "x_interp": (
                ("grid_point_hres", "weather_state"),
                np.array([[2.0], [4.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y": (
                ("grid_point_hres", "weather_state"),
                np.array([[1.0], [2.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("grid_point_hres", "weather_state"),
                np.array([[1.5], [3.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
        },
        coords={"grid_point_lres": [0, 1], "grid_point_hres": [0, 1], "weather_state": ["10u"]},
    )
    ds = ensure_member_zero_plot_variables(ds)
    assert "x_0" in ds
    assert "x_interp_0" in ds
    assert "y_0" in ds
    assert "y_pred_0" in ds
    np.testing.assert_allclose(get_plot_data_array(ds, "residuals_0").values[:, 0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(
        get_plot_data_array(ds, "residuals_pred_0").values[:, 0],
        np.array([0.5, 1.0], dtype=np.float32),
    )


def test_ensure_x_interp_for_plotting_reconstructs_missing_interp(tmp_path: Path, monkeypatch):
    pred_path = tmp_path / "run" / "predictions" / "predictions_20230829_step024.nc"
    pred_path.parent.mkdir(parents=True)
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("base\n", encoding="utf-8")
    ds = xr.Dataset(
        data_vars={
            "x": (
                ("grid_point_lres", "weather_state"),
                np.array([[1.0], [2.0]], dtype=np.float32),
                {"lon": "lon_lres", "lat": "lat_lres"},
            ),
            "y": (
                ("grid_point_hres", "weather_state"),
                np.array([[3.0], [4.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("grid_point_hres", "weather_state"),
                np.array([[5.0], [6.0]], dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_hres": (("grid_point_hres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([10.0, 11.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([10.0, 11.0], dtype=np.float32)),
        },
        coords={"grid_point_hres": [0, 1], "grid_point_lres": [0, 1], "weather_state": ["10u"]},
        attrs={"checkpoint_path": str(checkpoint_path)},
    )
    ds.to_netcdf(pred_path)

    class _FakeInterpolator:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def interpolate(self, x_values):
            return np.asarray([[7.0], [8.0]], dtype=np.float64)

    monkeypatch.setattr(
        "eval._backends.region_plotting.local_plotting.CheckpointResidualInterpolator",
        _FakeInterpolator,
    )

    with xr.open_dataset(pred_path) as ds_loaded:
        rebuilt = ensure_x_interp_for_plotting(ds_loaded, predictions_path=pred_path)

    assert "x_interp" in rebuilt
    np.testing.assert_allclose(rebuilt["x_interp"].values[:, 0], np.array([7.0, 8.0], dtype=np.float32))
    assert rebuilt["x_interp"].attrs["lon"] == "lon_hres"
    assert rebuilt["x_interp"].attrs["lat"] == "lat_hres"


def test_render_region_suite_from_predictions_file_writes_outputs(tmp_path: Path, monkeypatch):
    pred_path = tmp_path / "predictions_20230829_step024.nc"
    ds = xr.Dataset(
        data_vars={
            "x_interp": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 1, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 1, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 1, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_hres": (("grid_point_hres",), np.array([-80.0, -79.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([20.0, 21.0], dtype=np.float32)),
            "date": (("sample",), np.array(["2023-08-29T00:00:00"], dtype="datetime64[ns]")),
            "init_date": (("sample",), np.array(["2023-08-28T00:00:00"], dtype="datetime64[ns]")),
            "lead_step_hours": (("sample",), np.array([24], dtype=np.int32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [1],
            "grid_point_hres": [0, 1],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O1280"},
    )
    ds.to_netcdf(pred_path)

    def _fake_get_region_ds(ds_in, _region_box):
        return ds_in

    def _fake_plot_x_y(**_kwargs):
        return plt.figure()

    monkeypatch.setattr(mod, "get_region_ds", _fake_get_region_ds)
    monkeypatch.setattr(mod, "plot_x_y", _fake_plot_x_y)

    generated = mod.render_region_suite_from_predictions_file(
        predictions_nc=pred_path,
        out_dir=tmp_path / "plots",
        region_names=["idalia", "franklin"],
        model_variables=["x_interp", "y", "y_pred", "x_interp_minus_y_pred", "x_interp_minus_y"],
        also_png=True,
    )

    assert (tmp_path / "plots" / "all_regions_plots.pdf").exists()
    assert (tmp_path / "plots" / "idalia.pdf").exists()
    assert (tmp_path / "plots" / "idalia.png").exists()
    assert (tmp_path / "plots" / "franklin.pdf").exists()
    assert (tmp_path / "plots" / "franklin.png").exists()
    assert str(tmp_path / "plots" / "all_regions_plots.pdf") in generated


def test_render_region_suite_from_predictions_file_rebuilds_x_interp_before_selection(tmp_path: Path, monkeypatch):
    pred_path = tmp_path / "predictions_20230829_step024.nc"
    ds = xr.Dataset(
        data_vars={
            "x": (("sample", "grid_point_lres", "weather_state"), np.zeros((1, 2, 1), dtype=np.float32)),
            "y": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 1, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "y_pred": (
                ("sample", "ensemble_member", "grid_point_hres", "weather_state"),
                np.zeros((1, 1, 2, 1), dtype=np.float32),
                {"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_hres": (("grid_point_hres",), np.array([-80.0, -79.0], dtype=np.float32)),
            "lat_hres": (("grid_point_hres",), np.array([20.0, 21.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([-80.0, -79.0], dtype=np.float32)),
            "lat_lres": (("grid_point_lres",), np.array([20.0, 21.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [1],
            "grid_point_hres": [0, 1],
            "grid_point_lres": [0, 1],
            "weather_state": ["10u"],
        },
        attrs={"grid": "O1280"},
    )
    ds.to_netcdf(pred_path)

    captured: dict[str, list[str]] = {}

    def _fake_ensure(ds_in, **_kwargs):
        return ensure_member_zero_plot_variables(ds_in.assign(
            x_interp=xr.DataArray(
                np.zeros((2, 1), dtype=np.float32),
                dims=("grid_point_hres", "weather_state"),
                coords={"grid_point_hres": ds_in.coords["grid_point_hres"], "weather_state": ds_in.coords["weather_state"]},
                attrs={"lon": "lon_hres", "lat": "lat_hres"},
            )
        ))

    def _fake_get_region_ds(ds_in, _region_box):
        return ds_in

    def _fake_plot_x_y(**kwargs):
        captured["model_variables"] = kwargs["list_model_variables"]
        return plt.figure()

    monkeypatch.setattr(mod, "ensure_x_interp_for_plotting", _fake_ensure)
    monkeypatch.setattr(mod, "get_region_ds", _fake_get_region_ds)
    monkeypatch.setattr(mod, "plot_x_y", _fake_plot_x_y)

    mod.render_region_suite_from_predictions_file(
        predictions_nc=pred_path,
        out_dir=tmp_path / "plots",
        region_names=["idalia"],
        model_variables=["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"],
        also_png=False,
    )

    assert captured["model_variables"] == ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"]
