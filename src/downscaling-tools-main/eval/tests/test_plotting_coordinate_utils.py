from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eval._backends.region_plotting.plotting.coordinate_utils import get_region_ds


@pytest.fixture
def sample_region_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "field_hres": ("grid_point_hres", np.arange(4, dtype=np.float32)),
            "field_lres": ("grid_point_lres", np.arange(3, dtype=np.float32)),
            "lon_hres": ("grid_point_hres", np.array([-80.0, -60.0, -50.0, -40.0], dtype=np.float32)),
            "lat_hres": ("grid_point_hres", np.array([-20.0, -10.0, 0.0, 10.0], dtype=np.float32)),
            "lon_lres": ("grid_point_lres", np.array([-85.0, -55.0, -35.0], dtype=np.float32)),
            "lat_lres": ("grid_point_lres", np.array([-18.0, -5.0, 12.0], dtype=np.float32)),
        },
        coords={
            "grid_point_hres": np.arange(4),
            "grid_point_lres": np.arange(3),
        },
    )


def test_get_region_ds_subsets_named_region(sample_region_dataset: xr.Dataset) -> None:
    region_ds = get_region_ds(sample_region_dataset, "amazon_forest")

    assert region_ds.sizes["grid_point_hres"] == 2
    assert region_ds.sizes["grid_point_hres"] < sample_region_dataset.sizes["grid_point_hres"]
    assert region_ds.sizes["grid_point_lres"] == 1
    assert region_ds.sizes["grid_point_lres"] < sample_region_dataset.sizes["grid_point_lres"]
    assert region_ds.attrs["region"] == [-15.0, 5.0, -75.0, -45.0]


def test_get_region_ds_returns_empty_subset_for_empty_region(sample_region_dataset: xr.Dataset) -> None:
    region_ds = get_region_ds(sample_region_dataset, [60.0, 70.0, 60.0, 70.0])

    assert region_ds.sizes["grid_point_hres"] == 0
    assert region_ds.sizes["grid_point_lres"] == 0
    assert region_ds.attrs["region"] == [60.0, 70.0, 60.0, 70.0]
