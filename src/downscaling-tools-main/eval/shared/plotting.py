"""Shared plotting utilities used by multiple evaluators (TC, region_plot, intermediate)."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def get_coastlines():
    """Return the anemoi Coastlines renderer (singleton, lazy import).

    Works on plain matplotlib axes without cartopy projections.
    """
    from anemoi.training.diagnostics.maps import Coastlines

    return Coastlines()


def add_coastlines(ax, *, linewidth: float = 0.6) -> None:
    """Add coastline overlay to a matplotlib axes.

    Dispatches automatically:
    - If *ax* is a cartopy GeoAxes, uses ax.coastlines().
    - Otherwise, uses the anemoi Coastlines renderer (scatter-based, no projection needed).
    """
    if hasattr(ax, "coastlines"):
        ax.coastlines(linewidth=linewidth)
    else:
        get_coastlines().plot_continents(ax)


def select_projection(west: float, east: float, south: float, north: float):
    """Pick an appropriate cartopy projection for the given bounding box.

    Returns LambertConformal for non-dateline-crossing regions, PlateCarree otherwise.
    """
    from cartopy import crs

    crosses_dateline = east < west
    if crosses_dateline:
        return crs.PlateCarree()
    central_lon = (west + east) / 2.0
    central_lat = (south + north) / 2.0
    return crs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
