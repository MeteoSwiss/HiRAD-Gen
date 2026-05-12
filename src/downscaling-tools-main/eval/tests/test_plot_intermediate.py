from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from eval._backends.plot_intermediate.plot_intermediate import (
    plot_intermediate_trajectory,
    resolve_capture_steps,
    select_sampling_steps,
)


def test_select_sampling_steps_spreads_indices():
    steps = select_sampling_steps(total_steps=10, max_panels=4)
    assert steps[0] == 0
    assert steps[-1] == 9
    assert len(steps) == 4


def test_resolve_capture_steps_explicit_and_last_step():
    got = resolve_capture_steps(total_steps=20, explicit_steps=[0, 5, 10], capture_max_steps=0)
    assert got == [0, 5, 10, 19]


def test_resolve_capture_steps_with_cap():
    got = resolve_capture_steps(total_steps=20, explicit_steps=None, capture_max_steps=6)
    assert len(got) == 6
    assert got[0] == 0
    assert got[-1] == 19


def test_resolve_capture_steps_all_when_no_constraints():
    got = resolve_capture_steps(total_steps=5, explicit_steps=None, capture_max_steps=0)
    assert got == [0, 1, 2, 3, 4]


def test_resolve_capture_steps_invalid_explicit_raises():
    try:
        resolve_capture_steps(total_steps=5, explicit_steps=[10, 11], capture_max_steps=0)
    except ValueError as exc:
        assert "No valid capture steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range explicit steps")


def test_plot_intermediate_trajectory_writes_file(tmp_path: Path):
    n_steps = 5
    n_grid = 6
    ds = xr.Dataset(
        data_vars={
            "inter_state": (
                ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_steps, n_grid, 1).astype(np.float32),
            ),
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_grid, 1).astype(np.float32),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_grid, 1).astype(np.float32),
            ),
            "lon_hres": (["grid_point_hres"], np.array([0, 2, 4, 6, 8, 10], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([40, 42, 44, 46, 48, 50], dtype=np.float32)),
            "lon_lres": (["grid_point_lres"], np.array([0], dtype=np.float32)),
            "lat_lres": (["grid_point_lres"], np.array([45], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "sampling_step": np.arange(n_steps),
            "grid_point_hres": np.arange(n_grid),
            "grid_point_lres": [0],
            "weather_state": ["2t"],
        },
        attrs={"grid": "O320"},
    )

    out = tmp_path / "intermediate_plot.png"
    stats = tmp_path / "intermediate_plot_stats.json"
    result = plot_intermediate_trajectory(
        ds=ds,
        weather_state="2t",
        sample=0,
        member=0,
        out_path=str(out),
        region="default",
        max_panels=4,
        stats_out=str(stats),
    )

    assert result == out
    assert out.exists()
    assert stats.exists()
    with stats.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["weather_state"] == "2t"
    assert "final_step_checks" in payload


def test_plot_intermediate_trajectory_with_sparse_step_coords(tmp_path: Path):
    n_steps = 4
    n_grid = 6
    ds = xr.Dataset(
        data_vars={
            "inter_state": (
                ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_steps, n_grid, 1).astype(np.float32),
            ),
            "y_pred": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_grid, 1).astype(np.float32),
            ),
            "y": (
                ["sample", "ensemble_member", "grid_point_hres", "weather_state"],
                np.random.rand(1, 1, n_grid, 1).astype(np.float32),
            ),
            "lon_hres": (["grid_point_hres"], np.array([0, 2, 4, 6, 8, 10], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([40, 42, 44, 46, 48, 50], dtype=np.float32)),
            "lon_lres": (["grid_point_lres"], np.array([0], dtype=np.float32)),
            "lat_lres": (["grid_point_lres"], np.array([45], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "sampling_step": np.array([0, 3, 8, 19], dtype=np.int32),
            "grid_point_hres": np.arange(n_grid),
            "grid_point_lres": [0],
            "weather_state": ["2t"],
        },
        attrs={"grid": "O320"},
    )
    out = tmp_path / "intermediate_sparse.png"
    result = plot_intermediate_trajectory(
        ds=ds,
        weather_state="2t",
        sample=0,
        member=0,
        out_path=str(out),
        region="default",
        sampling_steps=[0, 19],
    )
    assert result == out
    assert out.exists()


def test_plot_intermediate_trajectory_missing_inter_state_raises():
    ds = xr.Dataset(
        data_vars={
            "y_pred": (["sample", "ensemble_member", "grid_point_hres", "weather_state"], np.zeros((1, 1, 2, 1), dtype=np.float32)),
            "lon_hres": (["grid_point_hres"], np.array([0, 1], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([40, 41], dtype=np.float32)),
            "lon_lres": (["grid_point_lres"], np.array([0], dtype=np.float32)),
            "lat_lres": (["grid_point_lres"], np.array([45], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "grid_point_hres": [0, 1],
            "grid_point_lres": [0],
            "weather_state": ["2t"],
        },
        attrs={"grid": "O320"},
    )
    try:
        plot_intermediate_trajectory(
            ds=ds,
            weather_state="2t",
            sample=0,
            member=0,
            out_path="/tmp/unused.png",
        )
    except ValueError as exc:
        assert "inter_state" in str(exc)
    else:
        raise AssertionError("Expected ValueError when inter_state is missing")


def test_plot_intermediate_trajectory_unknown_weather_state_raises(tmp_path: Path):
    ds = xr.Dataset(
        data_vars={
            "inter_state": (
                ["sample", "ensemble_member", "sampling_step", "grid_point_hres", "weather_state"],
                np.zeros((1, 1, 2, 2, 1), dtype=np.float32),
            ),
            "y_pred": (["sample", "ensemble_member", "grid_point_hres", "weather_state"], np.zeros((1, 1, 2, 1), dtype=np.float32)),
            "lon_hres": (["grid_point_hres"], np.array([0, 1], dtype=np.float32)),
            "lat_hres": (["grid_point_hres"], np.array([40, 41], dtype=np.float32)),
            "lon_lres": (["grid_point_lres"], np.array([0], dtype=np.float32)),
            "lat_lres": (["grid_point_lres"], np.array([45], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "sampling_step": [0, 1],
            "grid_point_hres": [0, 1],
            "grid_point_lres": [0],
            "weather_state": ["2t"],
        },
        attrs={"grid": "O320"},
    )
    try:
        plot_intermediate_trajectory(
            ds=ds,
            weather_state="msl",
            sample=0,
            member=0,
            out_path=str(tmp_path / "x.png"),
        )
    except ValueError as exc:
        assert "Unknown weather_state" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown weather state")
