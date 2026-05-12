"""Shared spatial utilities used by multiple evaluators (TC, region_plot)."""
from __future__ import annotations

import numpy as np


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    """Normalize longitude values to the range [-180, 180)."""
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def point_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    south: float,
    north: float,
    west: float,
    east: float,
) -> np.ndarray:
    """Boolean mask selecting points inside a lat/lon bounding box.

    Handles dateline-crossing boxes (where east < west after normalization).
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = normalize_lon(np.asarray(lon, dtype=np.float64))
    west_n = normalize_lon(np.asarray([west]))[0]
    east_n = normalize_lon(np.asarray([east]))[0]

    lat_mask = (lat_arr >= south) & (lat_arr <= north)
    if east_n >= west_n:
        lon_mask = (lon_arr >= west_n) & (lon_arr <= east_n)
    else:
        lon_mask = (lon_arr >= west_n) | (lon_arr <= east_n)
    return lat_mask & lon_mask
