"""Custom anemoi-datasets create sources for HiRAD-Gen.

`ifs-hres-files` builds a *trajectories* (forecast, 5D) anemoi dataset from the
per-(init, step) GRIBs written by ``ifs_hres_precompute.py``. The built-in
``grib`` source only implements ``execute_valid_dates`` and cannot serve a
trajectory build (which passes ``ForecastDates``); this source implements
``execute_forecast_dates``: for each (valid_time, basetime) pair it loads the
matching file ``ifs_hres_{init}_s{step}.grib`` and returns its fields with
date/time/step stamped from the basetime (mirroring ``ForcingsSource``).

Import this module before running ``anemoi.datasets.create`` tasks so the
source registers (see ``build_ifs_hres_anemoi.py``).
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from anemoi.transform.fields import new_field_with_metadata, new_fieldlist_from_list
from earthkit.data import from_source

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry


def init_step_filename(basetime, step_hours: int) -> str:
    """Match ifs_hres_precompute.out_name: init as YYMMDDHH, step as 3-digit hours."""
    return f"ifs_hres_{basetime.strftime('%y%m%d%H')}_s{step_hours:03d}.grib"


@source_registry.register("ifs-hres-files")
class IfsHresFilesSource(Source):
    """Trajectory source backed by precomputed per-(init, step) GRIB files."""

    def __init__(self, context: Any, directory: str, params: Optional[List[str]] = None) -> None:
        super().__init__(context)
        self.directory = directory
        self.params = params

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        result = []
        for valid_time, basetime in dates:
            step_hours = int((valid_time - basetime).total_seconds() // 3600)
            path = os.path.join(self.directory, init_step_filename(basetime, step_hours))
            fields = from_source("file", path).to_fieldlist()
            if self.params is not None:
                fields = fields.sel(param=self.params)
            meta = dict(
                date=int(basetime.strftime("%Y%m%d")),
                time=int(basetime.strftime("%H%M")),
                step=step_hours,
            )
            for f in fields:
                result.append(new_field_with_metadata(f, **meta))
        self.context.trace("📦", f"ifs-hres-files: {len(dates)} pairs → {len(result)} fields")
        return new_fieldlist_from_list(result)
