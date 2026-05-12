from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from eval._backends.region_plotting.plotting.datetime_utils import extract_date_from_dataset, safe_datetime_str


@pytest.fixture
def dataset_with_init_date() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "init_date": ("sample", np.array(["2023-08-26T00:00:00"], dtype="datetime64[ns]")),
        },
        coords={"sample": [0]},
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, ""),
        (None, ""),
        (np.nan, ""),
        (np.datetime64("NaT"), ""),
        (np.array([0]), ""),
        (np.int64(0), ""),
        (np.datetime64("2023-08-26"), "2023-08-26 00:00"),
        (pd.Timestamp("2023-08-26"), "2023-08-26 00:00"),
    ],
)
def test_safe_datetime_str_handles_valid_and_invalid_values(value: object, expected: str) -> None:
    assert safe_datetime_str(value) == expected


def test_extract_date_from_dataset_uses_init_date(dataset_with_init_date: xr.Dataset) -> None:
    assert extract_date_from_dataset(dataset_with_init_date) == "2023-08-26"


def test_extract_date_from_dataset_rejects_zero_epoch_placeholder() -> None:
    ds = xr.Dataset(
        data_vars={"date": ("sample", np.array([0], dtype=np.int64))},
        coords={"sample": [0]},
    )

    assert extract_date_from_dataset(ds) == ""


def test_extract_date_from_dataset_prefers_valid_init_date_over_bad_date() -> None:
    ds = xr.Dataset(
        data_vars={
            "init_date": ("sample", np.array(["2023-08-26T00:00:00"], dtype="datetime64[ns]")),
            "date": ("sample", np.array([0], dtype=np.int64)),
        },
        coords={"sample": [0]},
    )

    assert extract_date_from_dataset(ds) == "2023-08-26"
