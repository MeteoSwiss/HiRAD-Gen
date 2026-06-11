"""Safe date extraction and formatting for prediction datasets.

Fixes the 1970-01-01 bug where date=0 in NC files was interpreted as Unix epoch.
All plotters should use these functions instead of direct date access.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr


_EARLIEST_VALID_YEAR = 2000


def _first_scalar(value: Any) -> Any:
    """Return the first scalar value from array-like inputs."""
    if isinstance(value, xr.DataArray):
        value = value.values
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def is_valid_date(value: Any) -> bool:
    """Return True when *value* represents a real forecast date."""
    scalar = _first_scalar(value)
    if scalar is None:
        return False
    try:
        if isinstance(scalar, (int, float, np.integer, np.floating)):
            if scalar == 0 or np.isnan(float(scalar)):
                return False
        ts = pd.to_datetime(scalar)
        if pd.isna(ts):
            return False
        return ts.year >= _EARLIEST_VALID_YEAR
    except Exception:
        return False


def safe_datetime_str(value: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a date value safely, returning empty string on failure.

    Handles: None, NaN, NaT, integer 0 (epoch placeholder), numpy/pandas types.
    Returns empty string instead of "1970-01-01" for zero/null values.
    """
    scalar = _first_scalar(value)
    if scalar is None or not is_valid_date(scalar):
        return ""
    try:
        return pd.to_datetime(scalar).strftime(fmt)
    except Exception:
        return ""


def extract_date_from_dataset(
    ds: xr.Dataset,
    sample_idx: int = 0,
    key: str = "date",
    fallback_key: str = "init_date",
    fmt: str = "%Y-%m-%d",
) -> str:
    """Extract and format a date from an xarray dataset.

    Tries ``init_date`` first, then falls back to ``date``. Returns an empty
    string when no valid value is available.
    """
    for current_key in (fallback_key, key):
        if current_key not in ds:
            continue
        da = ds[current_key]
        if "sample" in da.dims:
            if not 0 <= sample_idx < int(da.sizes["sample"]):
                continue
            value = da.isel(sample=sample_idx).values
        else:
            value = da.values
        result = safe_datetime_str(value, fmt)
        if result:
            return result
    return ""
