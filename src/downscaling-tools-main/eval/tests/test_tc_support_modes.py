from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval._backends.tc import events as events_mod
from eval._backends.tc import workflows as workflows_mod
from eval._backends.tc import loading_grib as loading_grib_mod
from eval._backends.tc import loading_predictions as loading_pred_mod
from eval._backends.tc import data_types as data_types_mod
from eval._backends.tc import plot_config as plot_config_mod
from eval._backends.tc import experiment_config as exp_config_mod


def test_workflows_pdf_main_defaults_to_regridded(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(workflows_mod, "run_tc_pdf", _fake_run_tc_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflows.py",
            "pdf",
            "--predictions-dir",
            "/tmp/preds",
            "--outdir",
            "/tmp/out",
            "--run-label",
            "demo",
        ],
    )

    workflows_mod.main()

    assert captured["support_mode"] == "regridded"


def test_workflows_pdf_main_passes_display_label(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(workflows_mod, "run_tc_pdf", _fake_run_tc_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflows.py",
            "pdf",
            "--predictions-dir",
            "/tmp/preds",
            "--outdir",
            "/tmp/out",
            "--run-label",
            "manual_long_id",
            "--display-label",
            "short label",
        ],
    )

    workflows_mod.main()

    assert captured["display_label"] == "short label"


def test_regridded_prediction_curve_interpolates_gridded_source(tmp_path: Path):
    pred_path = tmp_path / "predictions_20230827_step024.nc"
    ds = xr.Dataset(
        data_vars={
            "y_pred": (
                ("sample", "ensemble_member", "lat_hres", "lon_hres", "weather_state"),
                np.array(
                    [
                        [
                            [
                                [[100000.0, 0.0, 0.0], [101000.0, 10.0, 0.0]],
                                [[102000.0, 20.0, 0.0], [103000.0, 30.0, 0.0]],
                            ]
                        ]
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "lat_hres": np.array([0.0, 1.0], dtype=np.float32),
            "lon_hres": np.array([0.0, 1.0], dtype=np.float32),
            "weather_state": np.array(["msl", "10u", "10v"], dtype=object),
        },
    )
    ds["y_pred"].attrs["lat"] = "lat_hres"
    ds["y_pred"].attrs["lon"] = "lon_hres"
    ds.to_netcdf(pred_path)

    curve = loading_pred_mod._load_prediction_curve_regridded(
        [(pred_path, 20230827, 24)],
        target_lon=np.array([0.5], dtype=np.float64),
        target_lat=np.array([0.5], dtype=np.float64),
    )

    assert curve.msl.shape == (1,)
    assert curve.wind.shape == (1,)
    assert np.isclose(curve.msl[0], 1015.0)
    assert np.isclose(curve.wind[0], 15.0)


def test_analysis_row_indices_supports_with_and_without_leading_analysis_frame():
    assert loading_grib_mod._analysis_row_indices(6, None) == slice(1, None)
    assert loading_grib_mod._analysis_row_indices(6, [0, 2]) == [1, 3]
    assert loading_grib_mod._analysis_row_indices(5, None) == slice(0, None)
    assert loading_grib_mod._analysis_row_indices(5, [0, 2]) == [0, 2]


def test_humberto_compute_event_stats_uses_runtime_oper_and_references():
    event = events_mod.EVENTS["humberto"]
    exp_cfg = exp_config_mod.TCExperimentConfig(
        analysis_expid="OPER_O96_0001",
        reference_expids=("ENFO_O48_0001",),
    )
    plot_cfg = plot_config_mod.PLOT_CONFIGS["humberto"]
    curves = {
        exp_cfg.analysis_expid: data_types_mod.CurveVectors(
            msl=np.array([999.0, 1001.0, 1003.0, 1005.0]),
            wind=np.array([12.0, 14.0, 16.0, 18.0]),
        ),
        exp_cfg.reference_expids[0]: data_types_mod.CurveVectors(
            msl=np.array([998.0, 1000.0, 1002.0, 1004.0]),
            wind=np.array([13.0, 15.0, 17.0, 19.0]),
        ),
        "demo-run": data_types_mod.CurveVectors(
            msl=np.array([996.0, 998.0, 1000.0, 1002.0]),
            wind=np.array([10.0, 12.0, 14.0, 16.0]),
        ),
    }

    stats = workflows_mod.compute_event_stats(
        curves,
        analysis_key=exp_cfg.analysis_expid,
        plot_config=plot_cfg,
        curve_order=["demo-run", *exp_cfg.reference_expids],
    )

    assert stats["curve_order"] == ["demo-run", *exp_cfg.reference_expids]
    assert exp_cfg.reference_expids[0] in stats["variables"]["mslp_hpa"]["curves"]
