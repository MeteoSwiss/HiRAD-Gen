"""GRIB file I/O for TC evaluation."""
from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Iterable

import earthkit.data as ekd
import numpy as np
import xarray as xr

from .data_types import BoundingBox, CurveVectors, FORECAST_STEP_COUNT, SupportMode, step_to_index
from .grid import normalize_lon


def _import_metview():
    import metview as mv  # type: ignore

    return mv


def _analysis_row_indices(frame_count: int, step_indices: list[int] | None) -> slice | list[int]:
    offset = 1 if frame_count > FORECAST_STEP_COUNT else 0
    if step_indices is None:
        return slice(offset, None)
    return [offset + idx for idx in step_indices]


def regridded_target_points(
    bbox: BoundingBox,
    regrid_resolution: float,
    sample_grib_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the regular target grid for regridded mode from a sample GRIB."""
    mv = _import_metview()
    field = mv.read(
        data=mv.read(sample_grib_path),
        grid=[regrid_resolution, regrid_resolution],
        area=[bbox.south, bbox.west, bbox.north, bbox.east],
        param="msl",
    )
    ds = field.to_dataset()
    latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
    longitudes = normalize_lon(np.asarray(ds["longitude"].values, dtype=np.float64))
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    return lon_grid.reshape(-1), lat_grid.reshape(-1)


def load_grib_curves(
    *,
    dir_data_base: str,
    event_name: str,
    analysis_expid: str,
    analysis_dates: list[str],
    forecast_dates: list[str],
    reference_expids: Iterable[str] = (),
    support_mode: SupportMode = "regridded",
    bbox: BoundingBox | None = None,
    regrid_resolution: float = 0.25,
    steps: Iterable[int] | None = None,
    max_pf_members: int | None = None,
) -> dict[str, CurveVectors]:
    """Load analysis + reference curves from GRIB files.

    Returns dict keyed by expid (e.g. {"OPER_O320_0001": ..., "ENFO_O320_0001": ...}).
    """
    step_indices = None if steps is None else [step_to_index(step) for step in steps]
    expids = list(dict.fromkeys(reference_expids))
    event_dir = Path(dir_data_base) / event_name

    if support_mode == "native":
        curves: dict[str, CurveVectors] = {}
        curves[analysis_expid] = _load_native_curve(
            [str(event_dir / f"surface_an_{analysis_expid}_{date}.grib") for date in analysis_dates],
            is_analysis=True,
            step_indices=step_indices,
        )
        for expid in expids:
            curves[expid] = _load_native_curve(
                [str(event_dir / f"surface_pf_{expid}_{date}.grib") for date in forecast_dates],
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    if support_mode == "regridded":
        if bbox is None:
            raise ValueError("bbox is required for regridded support mode")
        curves = {}
        curves[analysis_expid] = _load_regridded_curve(
            [str(event_dir / f"surface_an_{analysis_expid}_{date}.grib") for date in analysis_dates],
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=True,
            step_indices=step_indices,
        )
        for expid in expids:
            curves[expid] = _load_regridded_curve(
                [str(event_dir / f"surface_pf_{expid}_{date}.grib") for date in forecast_dates],
                bbox=bbox,
                regrid_resolution=regrid_resolution,
                is_analysis=False,
                step_indices=step_indices,
                max_pf_members=max_pf_members,
            )
        return curves

    raise ValueError(f"Unsupported support_mode={support_mode!r}")


def load_grib_curve_from_paths(
    grib_paths: list[str] | str,
    *,
    support_mode: SupportMode = "native",
    bbox: BoundingBox | None = None,
    regrid_resolution: float = 0.25,
) -> CurveVectors:
    """Load MSL + 10m wind from arbitrary GRIB files (IEKM, ENFO, or any custom reference).

    grib_paths may be a single string with glob wildcards or a list of paths.
    """
    if isinstance(grib_paths, str):
        if "*" in grib_paths or "?" in grib_paths:
            files = sorted(_glob.glob(grib_paths))
        else:
            files = [grib_paths]
    else:
        files = list(grib_paths)
    if not files:
        raise FileNotFoundError(f"No GRIB files matched: {grib_paths}")

    if support_mode == "native":
        return _load_native_curve(files, is_analysis=False, step_indices=None)

    if support_mode == "regridded":
        if bbox is None:
            raise ValueError("bbox is required for regridded support mode")
        return _load_regridded_curve(
            files,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=False,
            step_indices=None,
        )

    raise ValueError(f"Unsupported support_mode={support_mode!r}")


# --- Internal helpers ---


def _load_native_curve(
    files: list[str],
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
) -> CurveVectors:
    ds = ekd.from_source("file", files).to_xarray(engine="cfgrib")
    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=normalize_lon(ds["longitude"].values.astype(np.float64)))
    return CurveVectors(
        msl=_extract_native_values(
            ds["msl"],
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
            scale=0.01,
        ),
        wind=_extract_native_wind(
            ds,
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
    )


def _extract_native_values(
    da: xr.DataArray,
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
    scale: float = 1.0,
) -> np.ndarray:
    if "number" in da.dims and int(da.sizes["number"]) > 1 and max_pf_members is not None:
        da = da.isel(number=slice(0, max_pf_members))

    if is_analysis:
        if "forecast_reference_time" in da.dims:
            if step_indices is None:
                da = da.isel(forecast_reference_time=slice(1, None))
            else:
                da = da.isel(forecast_reference_time=[1 + idx for idx in step_indices])
    elif step_indices is not None and "step" in da.dims:
        da = da.isel(step=step_indices)

    return (np.asarray(da.values, dtype=np.float64) * scale).reshape(-1)


def _extract_native_wind(
    ds: xr.Dataset,
    *,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    u10 = _extract_native_values(
        ds["u10"],
        is_analysis=is_analysis,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    v10 = _extract_native_values(
        ds["v10"],
        is_analysis=is_analysis,
        step_indices=step_indices,
        max_pf_members=max_pf_members,
    )
    return np.sqrt(u10 * u10 + v10 * v10)


def _load_regridded_curve(
    files: list[str],
    *,
    bbox: BoundingBox,
    regrid_resolution: float,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None = None,
) -> CurveVectors:
    mv = _import_metview()
    holders = [mv.read(path) for path in files]
    return CurveVectors(
        msl=_extract_regridded_values(
            holders,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            parameter="msl",
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
        wind=_extract_regridded_wind(
            holders,
            bbox=bbox,
            regrid_resolution=regrid_resolution,
            is_analysis=is_analysis,
            step_indices=step_indices,
            max_pf_members=max_pf_members,
        ),
    )


def _read_regridded_variable(
    data, *, bbox: BoundingBox, regrid_resolution: float, parameter: str
) -> np.ndarray:
    mv = _import_metview()
    field = mv.read(
        data=data,
        grid=[regrid_resolution, regrid_resolution],
        area=[bbox.south, bbox.west, bbox.north, bbox.east],
        param=parameter,
    )
    dataset = field.to_dataset()
    if parameter == "msl":
        return np.asarray(dataset["msl"].values, dtype=np.float64) / 100.0
    if parameter == "10u":
        return np.asarray(dataset["u10"].values, dtype=np.float64)
    if parameter == "10v":
        return np.asarray(dataset["v10"].values, dtype=np.float64)
    raise ValueError(f"Unsupported parameter={parameter!r}")


def _extract_regridded_values(
    holders: list,
    *,
    bbox: BoundingBox,
    regrid_resolution: float,
    parameter: str,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for holder in holders:
        arr = _read_regridded_variable(holder, bbox=bbox, regrid_resolution=regrid_resolution, parameter=parameter)
        if is_analysis:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        else:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)


def _extract_regridded_wind(
    holders: list,
    *,
    bbox: BoundingBox,
    regrid_resolution: float,
    is_analysis: bool,
    step_indices: list[int] | None,
    max_pf_members: int | None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for holder in holders:
        u10 = _read_regridded_variable(holder, bbox=bbox, regrid_resolution=regrid_resolution, parameter="10u")
        v10 = _read_regridded_variable(holder, bbox=bbox, regrid_resolution=regrid_resolution, parameter="10v")
        arr = np.sqrt(u10 * u10 + v10 * v10)
        if is_analysis:
            arr = arr[_analysis_row_indices(arr.shape[0], step_indices), :, :]
        else:
            if max_pf_members is not None:
                arr = arr[:max_pf_members, :, :, :]
            if step_indices is not None:
                arr = arr[:, step_indices, :, :]
        chunks.append(arr.reshape(-1))
    return np.concatenate(chunks)
