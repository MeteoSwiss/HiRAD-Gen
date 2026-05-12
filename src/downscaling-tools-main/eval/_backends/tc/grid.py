"""Spatial operations — pure numpy/xarray, no I/O."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .data_types import BoundingBox, StructuredGrid


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def point_mask(lon: np.ndarray, lat: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Boolean mask selecting points inside bbox, with dateline handling."""
    lat_mask = (lat >= bbox.south) & (lat <= bbox.north)
    west = normalize_lon(np.asarray([bbox.west], dtype=np.float64))[0]
    east = normalize_lon(np.asarray([bbox.east], dtype=np.float64))[0]
    if east >= west:
        lon_mask = (lon >= west) & (lon <= east)
    else:
        lon_mask = (lon >= west) | (lon <= east)
    return lat_mask & lon_mask


def structured_grid_from_points(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    required: bool = True,
) -> StructuredGrid | None:
    lon = normalize_lon(np.asarray(lon, dtype=np.float64)).reshape(-1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    if lon.shape != lat.shape:
        raise ValueError("lon/lat point arrays must have identical flattened size")

    lon_axis = np.unique(lon)
    lat_axis = np.unique(lat)
    if lon_axis.size * lat_axis.size != lon.size:
        if required:
            raise ValueError("Point set does not describe a structured lat/lon grid")
        return None

    lon_index = np.searchsorted(lon_axis, lon)
    lat_index = np.searchsorted(lat_axis, lat)
    point_indices = np.full((lat_axis.size, lon_axis.size), -1, dtype=np.int64)
    for flat_index, (iy, ix) in enumerate(zip(lat_index, lon_index)):
        if point_indices[iy, ix] != -1:
            if required:
                raise ValueError("Duplicate source points prevent structured-grid reconstruction")
            return None
        point_indices[iy, ix] = flat_index

    if np.any(point_indices < 0):
        if required:
            raise ValueError("Missing source points prevent structured-grid reconstruction")
        return None

    return StructuredGrid(
        lat_axis=np.asarray(lat_axis, dtype=np.float64),
        lon_axis=np.asarray(lon_axis, dtype=np.float64),
        point_indices=point_indices,
    )


def interp_structured(
    values_by_point: np.ndarray,
    *,
    src_grid: StructuredGrid,
    target_grid: StructuredGrid,
) -> np.ndarray:
    source = np.asarray(values_by_point, dtype=np.float64)[:, src_grid.point_indices]
    data = xr.DataArray(
        source,
        dims=("member", "lat", "lon"),
        coords={"lat": src_grid.lat_axis, "lon": src_grid.lon_axis},
    )
    interpolated = data.interp(
        lat=target_grid.lat_axis,
        lon=target_grid.lon_axis,
        method="linear",
    )
    return np.asarray(interpolated.values, dtype=np.float64)


def nearest_point_indices(
    *,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.column_stack([src_lon, src_lat]))
        _, idx = tree.query(np.column_stack([target_lon, target_lat]), k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        idx = np.empty(target_lon.shape[0], dtype=np.int64)
        for i, (lon, lat) in enumerate(zip(target_lon, target_lat)):
            idx[i] = int(np.argmin((src_lon - lon) ** 2 + (src_lat - lat) ** 2))
        return idx
