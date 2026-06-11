"""Unified prediction file discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Canonical prediction filename: predictions_YYYYMMDD_stepNNN.nc
# Written by eval/predict/output_writer.py as:
#   f"predictions_{date}_step{int(step):03d}.nc"
# Used by: tc/loading_predictions.py, jobs/materialize_x_interp_reference.py,
#          weight_diagnostics/mechanistic_compare_v1.py,
#          jobs/select_tc_representative_case.py,
#          jobs/compare_proxy_tc_extremes.py,
#          jobs/templates/stage_prediction_spectra_gribs.py,
#          region_plotting/plot_one_date_local.py (glob variant)
PREDICTION_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    date: str     # YYYYMMDD
    step: int     # lead time in hours
    member: int   # always 0 — members are inside the NetCDF, not in the filename


def find_predictions(predictions_dir: Path) -> list[PredictionFile]:
    """Scan directory for prediction NetCDFs.

    Returns list of PredictionFile sorted by (date, step, member).
    Handles the canonical filename pattern: predictions_YYYYMMDD_stepNNN.nc
    """
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.is_dir():
        return []
    results: list[PredictionFile] = []
    for path in sorted(predictions_dir.glob("predictions_*.nc")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        results.append(PredictionFile(
            path=path,
            date=match.group(1),
            step=int(match.group(2)),
            member=0,
        ))
    results.sort(key=lambda p: (p.date, p.step, p.member))
    return results


def discover_shape(predictions_dir: Path) -> dict:
    """Extract available members, steps, dates from a predictions directory.

    Returns: {"members": [...], "steps": [...], "dates": [...]}
    Members are always [0] since member info lives inside NetCDF files.
    """
    preds = find_predictions(predictions_dir)
    dates = sorted({p.date for p in preds})
    steps = sorted({p.step for p in preds})
    members = sorted({p.member for p in preds})
    return {"members": members, "steps": steps, "dates": dates}
