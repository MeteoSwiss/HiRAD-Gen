"""Plot metadata extraction and title helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .datetime_utils import safe_datetime_str


@dataclass
class PlotMetadata:
    """Common plot metadata used to build region plot titles."""

    region: str
    date: str = ""
    init_date: str = ""
    lead_hours: int | None = None
    sample_position: int = 0
    ensemble_member: int = 0

    def to_title(self, sep: str = " | ") -> str:
        """Build a standard title string."""
        parts = [self.region]
        if self.date:
            parts.append(f"date={self.date}")
        if self.init_date:
            parts.append(f"init={self.init_date}")
        if self.lead_hours is not None:
            parts.append(f"step={self.lead_hours}h")
        parts.append(f"sample_pos={self.sample_position}")
        return sep.join(parts)


def _scalar_value(ds: xr.Dataset, name: str, sample_idx: int = 0) -> Any | None:
    """Extract a scalar-like dataset value."""
    if name not in ds:
        return None
    value = ds[name]
    if "sample" in value.dims:
        if not 0 <= sample_idx < int(value.sizes["sample"]):
            return None
        raw = value.isel(sample=sample_idx).values
    else:
        raw = value.values
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return None
        raw = raw.reshape(-1)[0]
    if hasattr(raw, "item") and not isinstance(raw, (str, bytes)):
        try:
            raw = raw.item()
        except Exception:
            pass
    return raw


def extract_plot_metadata(
    ds: xr.Dataset,
    region_name: str,
    sample_idx: int = 0,
    ensemble_member_idx: int = 0,
) -> PlotMetadata:
    """Extract standard plotting metadata from a dataset."""
    lead_hours: int | None = None
    lead_value = _scalar_value(ds, "lead_step_hours", sample_idx)
    if lead_value is not None:
        try:
            lead_hours = int(lead_value)
        except Exception:
            lead_hours = None

    return PlotMetadata(
        region=region_name,
        date=safe_datetime_str(_scalar_value(ds, "date", sample_idx)),
        init_date=safe_datetime_str(_scalar_value(ds, "init_date", sample_idx)),
        lead_hours=lead_hours,
        sample_position=sample_idx,
        ensemble_member=ensemble_member_idx,
    )


def sample_meta_title(ds_region: xr.Dataset, region_name: str, sample_idx: int = 0) -> str:
    """Build the batch-plot title used by ``plot_regions.py``."""
    metadata = extract_plot_metadata(ds_region, region_name, sample_idx=sample_idx)
    parts = [region_name, f"sample_pos={sample_idx}"]
    sample_value = _scalar_value(ds_region, "sample", sample_idx)
    if sample_value is not None:
        try:
            parts.append(f"sample_id={int(sample_value)}")
        except Exception:
            pass
    if metadata.date:
        parts.append(f"date={metadata.date}")
    if metadata.init_date:
        parts.append(f"init={metadata.init_date}")
    if metadata.lead_hours is not None:
        parts.append(f"lead_h={metadata.lead_hours}")
    return " | ".join(parts)


def step_meta_title(region_name: str, step: Any, forecast_ref_time: Any) -> str:
    """Build the legacy non-sample title used by ``plot_regions.py``."""
    parts = [f"{region_name}", f"step={step}"]
    forecast_ref = safe_datetime_str(forecast_ref_time)
    if forecast_ref:
        parts.append(f"forecast_ref={forecast_ref}")
    return " | ".join(parts)


def build_run_plot_title(
    ds_member: xr.Dataset,
    *,
    region: str,
    run_label: str,
    sample_index: int,
    member_label: str,
    ensemble_member_index: int = 0,
) -> str:
    """Build the single-date plot title used by ``plot_one_date_local.py``."""
    metadata = extract_plot_metadata(
        ds_member,
        region,
        sample_idx=sample_index,
        ensemble_member_idx=ensemble_member_index,
    )
    parts: list[str] = [region, f"sample_pos={sample_index}"]
    if run_label:
        parts.append(f"run={run_label}")
    if metadata.date:
        parts.append(f"date={metadata.date}")
    if metadata.init_date:
        parts.append(f"init={metadata.init_date}")
    if metadata.lead_hours is not None:
        parts.append(f"lead_h={metadata.lead_hours}")
    if member_label:
        parts.append(member_label)
    return " | ".join(parts)
