from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from manual_inference.config import BUNDLE_IMPLICIT_HRES_FEATURES
from manual_inference.config import DEFAULT_CONSTANT_FORCINGS_NPZ
from manual_inference.config import DEFAULT_LRES_PL_CHANNELS
from manual_inference.config import DEFAULT_LRES_SFC_CHANNELS
from manual_inference.config import DEFAULT_TARGET_PL_CHANNELS
from manual_inference.config import DEFAULT_TARGET_SFC_CHANNELS
from manual_inference.config import FALLBACK_CONSTANT_FORCINGS_NPZ
from manual_inference.config import OPTIONAL_ZERO_LRES_SFC_VARS
from manual_inference.config import SFC_TO_CFGRIB

# Re-export for backward compatibility — external callers import these from here.
__all__ = [
    "BUNDLE_IMPLICIT_HRES_FEATURES",
    "DEFAULT_CONSTANT_FORCINGS_NPZ",
    "FALLBACK_CONSTANT_FORCINGS_NPZ",
    "OPTIONAL_ZERO_LRES_SFC_VARS",
    "SFC_TO_CFGRIB",
]


def split_level_channel(name: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_([0-9]+)$", name)
    if match is None:
        return name, None
    return match.group(1), int(match.group(2))


def _cleanup_empty_cfgrib_indexes(grib_path: str | Path) -> None:
    path = Path(grib_path)
    for idx_path in path.parent.glob(f"{path.name}.*.idx"):
        try:
            if idx_path.is_file() and idx_path.stat().st_size == 0:
                idx_path.unlink()
        except FileNotFoundError:
            continue


def _open_cfgrib_dataset(
    grib_path: str | Path,
    *,
    filter_by_keys: Mapping[str, str] | None = None,
) -> xr.Dataset:
    _cleanup_empty_cfgrib_indexes(grib_path)
    backend_kwargs = {}
    if filter_by_keys:
        backend_kwargs["filter_by_keys"] = dict(filter_by_keys)
    try:
        return xr.open_dataset(str(grib_path), engine="cfgrib", backend_kwargs=backend_kwargs)
    except Exception:
        import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

        return ekd.from_source("file", str(grib_path)).to_xarray(
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
        )


def _normalize_optional_grib_list(
    grib_paths: Sequence[str | Path] | None,
) -> list[Path]:
    normalized: list[Path] = []
    for raw in grib_paths or ():
        raw_text = str(raw).strip()
        if not raw_text:
            continue
        normalized.append(Path(raw_text))
    return normalized


def _merge_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    if not datasets:
        raise ValueError("Expected at least one dataset to merge.")
    merged = datasets[0]
    for ds in datasets[1:]:
        merged = xr.merge(
            [merged, ds],
            compat="override",
            join="exact",
            combine_attrs="override",
        )
    return merged


def _open_selected_surface_dataset(
    primary_grib: str | Path,
    extra_gribs: Sequence[str | Path] | None,
    *,
    step_hours: int | None,
    member: int | None,
    filter_by_keys: Mapping[str, str] | None = None,
    allow_missing_member: bool = False,
) -> xr.Dataset:
    datasets = []
    for path in [Path(primary_grib), *_normalize_optional_grib_list(extra_gribs)]:
        ds = _open_cfgrib_dataset(path, filter_by_keys=filter_by_keys)
        ds = _select_step(ds, step_hours)
        ds = _select_member(ds, member, allow_missing=allow_missing_member)
        datasets.append(ds)
    return _merge_datasets(datasets)


def _infer_target_gribs_from_hres(
    hres_grib: str | Path,
) -> tuple[Path | None, Path | None]:
    """Try to infer colocated target GRIB paths from hres source file name.

    Example:
      enfo_o320_0001_date20230829_time0000_step24to120_sfc.grib
    -> enfo_o320_0001_date20230829_time0000_mem1to10_step24to120_sfc_y.grib
       enfo_o320_0001_date20230829_time0000_mem1to10_step24to120_pl_y.grib

      enfo_o96_0001_date20250926_time0000_step24to120_sfc.grib
    -> iekm_o96_iekm_date20250926_time0000_step24to120_sfc_y.grib
       iekm_o96_iekm_date20250926_time0000_step24to120_pl_y.grib
    """
    hres_path = Path(hres_grib)
    name = hres_path.name
    candidate_pairs: list[tuple[str, str]] = []

    m = re.match(
        r"^(enfo_o320_[^_]+_date\d{8}_time\d{4})_(step[^_]+)_sfc\.grib$",
        name,
    )
    if m is not None:
        prefix = m.group(1)
        step_token = m.group(2)
        candidate_pairs.append(
            (
                f"{prefix}_mem1to10_{step_token}_sfc_y.grib",
                f"{prefix}_mem1to10_{step_token}_pl_y.grib",
            )
        )

    m = re.match(
        r"^enfo_o96_[^_]+_date(\d{8})_time(\d{4})_(step[^_]+)_sfc\.grib$",
        name,
    )
    if m is not None:
        date_token = m.group(1)
        time_token = m.group(2)
        step_token = m.group(3)
        candidate_pairs.append(
            (
                f"iekm_o96_iekm_date{date_token}_time{time_token}_{step_token}_sfc_y.grib",
                f"iekm_o96_iekm_date{date_token}_time{time_token}_{step_token}_pl_y.grib",
            )
        )

    if not candidate_pairs:
        return None, None

    search_roots: list[Path] = [hres_path.parent]
    if hres_path.parent != hres_path.parent.parent:
        search_roots.append(hres_path.parent.parent)

    def _resolve(filename: str) -> Path | None:
        for root in search_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    for sfc_name, pl_name in candidate_pairs:
        sfc_path = _resolve(sfc_name)
        pl_path = _resolve(pl_name)
        if sfc_path is not None or pl_path is not None:
            return sfc_path, pl_path
    return None, None


def parse_valid_time(value, fallback) -> datetime:
    raw = value if value is not None else fallback
    if raw is None or str(raw).strip().lower() == "unknown":
        raise ValueError(
            "No valid time available. Provide `valid_time_override` or ensure GRIB contains `valid_time`."
        )
    return datetime.fromisoformat(str(raw).replace("Z", "").split(".")[0])


def _open_bundle_dataset(path: str | Path) -> xr.Dataset:
    bundle_path = Path(path)
    if bundle_path.suffix == ".zarr" or bundle_path.is_dir():
        return xr.open_zarr(bundle_path, consolidated=False)
    return xr.open_dataset(bundle_path)


def _interpolate_level_points(
    field: xr.DataArray,
    available_levels: Sequence[int],
    target_level: int,
) -> np.ndarray:
    levels_np = np.asarray(available_levels, dtype=np.int32)
    order = np.argsort(levels_np)
    sorted_levels = levels_np[order]
    if target_level < int(sorted_levels[0]) or target_level > int(sorted_levels[-1]):
        raise KeyError(f"Missing level {target_level}")
    if "level" not in field.dims:
        raise KeyError("Missing 'level' dimension for pressure-level interpolation")
    values = np.asarray(field.transpose("level", ...).values, dtype=np.float32).reshape(len(levels_np), -1)
    values = values[order, :]
    interp = np.empty(values.shape[1], dtype=np.float32)
    for idx in range(values.shape[1]):
        interp[idx] = np.interp(float(target_level), sorted_levels, values[:, idx])
    return interp


def open_bundle_dataset(path: str | Path) -> xr.Dataset:
    return _open_bundle_dataset(path)


def _borrow_or_open_bundle_dataset(
    bundle_nc: str | Path | xr.Dataset,
) -> tuple[xr.Dataset, bool]:
    if isinstance(bundle_nc, xr.Dataset):
        return bundle_nc, False
    return open_bundle_dataset(bundle_nc), True


def _extract_explicit_hres_input_channel(bundle: xr.Dataset, name: str) -> np.ndarray | None:
    n_points = int(bundle.sizes["point_hres"])
    base, level = split_level_channel(name)
    candidate_names = [f"in_hres_{name}"]
    if base != name:
        candidate_names.append(f"in_hres_{base}")

    for var_name in candidate_names:
        if var_name not in bundle:
            continue
        da = bundle[var_name]
        if level is not None:
            selected = False
            for lev_name in ("level", "target_level", "isobaricInhPa"):
                if lev_name not in da.coords:
                    continue
                levels = np.asarray(da[lev_name].values).astype(int).tolist()
                if int(level) not in levels:
                    break
                da = da.sel({lev_name: int(level)})
                selected = True
                break
            if not selected and any(n in da.coords for n in ("level", "target_level", "isobaricInhPa")):
                continue

        ok = True
        for dim in list(da.dims):
            if dim == "point_hres":
                continue
            if da.sizes[dim] != 1:
                ok = False
                break
            da = da.isel({dim: 0})
        if not ok:
            continue

        vals = np.asarray(da.values, dtype=np.float32).reshape(-1)
        if vals.size != n_points:
            continue
        return vals
    return None


def find_missing_explicit_hres_inputs(
    bundle_nc: str | Path | xr.Dataset,
    requested_hres_names: Sequence[str],
) -> list[str]:
    bundle, should_close = _borrow_or_open_bundle_dataset(bundle_nc)
    try:
        missing: list[str] = []
        for name in requested_hres_names:
            if name in BUNDLE_IMPLICIT_HRES_FEATURES:
                continue
            if _extract_explicit_hres_input_channel(bundle, name) is None:
                missing.append(name)
        return missing
    finally:
        if should_close:
            try:
                bundle.close()
            except Exception:
                pass


def _get_pl_level_coord(ds_pl: xr.Dataset) -> str:
    for name in ds_pl.coords:
        if "isobaric" in name.lower() or name.lower() == "level":
            return name
    raise KeyError("No pressure-level coordinate found in PL GRIB dataset")


def _extract_sfc_field(ds_sfc: xr.Dataset, channel_name: str) -> np.ndarray:
    key = SFC_TO_CFGRIB.get(channel_name, channel_name)
    if key not in ds_sfc:
        raise KeyError(f"Missing SFC variable for channel {channel_name} (expected {key})")
    return np.asarray(ds_sfc[key].values, dtype=np.float32).squeeze()


def _extract_pl_field(ds_pl: xr.Dataset, base: str, level: int) -> np.ndarray:
    if base not in ds_pl:
        raise KeyError(f"Missing PL variable: {base}")
    level_coord = _get_pl_level_coord(ds_pl)
    levels = np.asarray(ds_pl[level_coord].values).astype(int).tolist()
    if int(level) not in levels:
        raise KeyError(f"Missing PL level {level} for variable {base}")
    return np.asarray(ds_pl[base].sel({level_coord: int(level)}).values, dtype=np.float32).squeeze()


def _normalize_channel_subset(
    channels: Sequence[str] | None,
    *,
    default: Sequence[str],
) -> list[str]:
    if channels is None:
        return list(default)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in channels:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def parse_channel_subset_csv(raw: str) -> list[str] | None:
    """Parse a CSV channel string with explicit-empty sentinel support.

    Returns ``None`` for empty/whitespace input (meaning "use defaults"),
    ``[]`` for sentinel values ``NONE``, ``empty``, ``-``, ``[]``
    (meaning "explicitly no channels"), or a list of stripped channel names.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.lower() in {"none", "empty", "-", "[]"}:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Keep underscore alias so in-module callers and tests using the old name
# continue to work without a separate migration.
_parse_channel_subset_csv = parse_channel_subset_csv


def _load_constant_forcings_for_size(
    npz_candidates: Sequence[str | Path],
    target_size: int,
) -> tuple[dict[str, np.ndarray], str | None, list[str]]:
    constant_values: dict[str, np.ndarray] = {}
    tried_paths: list[str] = []

    for candidate in npz_candidates:
        npz_path = Path(candidate)
        if not npz_path.exists():
            continue
        tried_paths.append(str(npz_path))
        with np.load(npz_path) as npz:
            current: dict[str, np.ndarray] = {}
            for key in npz.files:
                arr = np.asarray(npz[key], dtype=np.float32).reshape(-1)
                if arr.size == target_size:
                    current[key] = arr
        if current:
            constant_values = current
            return constant_values, str(npz_path), tried_paths

    return constant_values, None, tried_paths


def load_lres_fields_from_grib(
    sfc_grib: str | Path,
    pl_grib: str | Path,
    lres_channel_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    ds_sfc = _open_cfgrib_dataset(sfc_grib)
    ds_pl = _open_cfgrib_dataset(pl_grib)
    fields: dict[str, np.ndarray] = {}
    for name in lres_channel_names:
        base, level = split_level_channel(name)
        if level is None:
            fields[name] = _extract_sfc_field(ds_sfc, name)
        else:
            fields[name] = _extract_pl_field(ds_pl, base, level)
    valid_time = ds_sfc.get("valid_time")
    valid_time = None if valid_time is None else np.asarray(valid_time.values).squeeze()
    return fields, valid_time


def read_hres_static_from_grib(hres_grib: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel

    ds_hr = _open_cfgrib_dataset(hres_grib)
    z = np.asarray(ds_hr["z"].values, dtype=np.float32).squeeze() if "z" in ds_hr else None
    lsm = np.asarray(ds_hr["lsm"].values, dtype=np.float32).squeeze() if "lsm" in ds_hr else None
    return z, lsm


def fill_hres_features(
    x_hres,
    name_to_idx_hres: Mapping[str, int],
    lat_hres: np.ndarray,
    lon_hres: np.ndarray,
    dt: datetime,
    device,
    *,
    z: np.ndarray | None = None,
    lsm: np.ndarray | None = None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> None:
    import earthkit.data as ekd  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from anemoi.transform.grids.unstructured import (
        UnstructuredGridFieldList,
    )  # pylint: disable=import-outside-toplevel

    constant_values: dict[str, np.ndarray] = {}
    constant_source: str | None = None
    tried_constant_npz: list[str] = []
    npz_candidates: list[str | Path] = []
    if constant_forcings_npz is not None:
        npz_candidates.append(constant_forcings_npz)
        for fallback in FALLBACK_CONSTANT_FORCINGS_NPZ:
            if str(fallback) != str(constant_forcings_npz):
                npz_candidates.append(fallback)
        constant_values, constant_source, tried_constant_npz = _load_constant_forcings_for_size(
            npz_candidates, lat_hres.size
        )

    # Prefer NPZ constants for strict parity with runner constant forcings.
    constant_features = (
        "z",
        "lsm",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
    )
    for name in constant_features:
        if name not in name_to_idx_hres:
            continue
        if name in constant_values:
            arr = constant_values[name]
        elif name == "z" and z is not None:
            arr = np.asarray(z, dtype=np.float32).reshape(-1)
        elif name == "lsm" and lsm is not None:
            arr = np.asarray(lsm, dtype=np.float32).reshape(-1)
        elif name in {"cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude"}:
            lat_rad = np.deg2rad(lat_hres.astype(np.float64))
            lon_rad = np.deg2rad(lon_hres.astype(np.float64))
            geometric = {
                "cos_latitude": np.cos(lat_rad),
                "sin_latitude": np.sin(lat_rad),
                "cos_longitude": np.cos(lon_rad),
                "sin_longitude": np.sin(lon_rad),
            }
            arr = geometric[name].astype(np.float32)
        else:
            tried = ", ".join(tried_constant_npz) if tried_constant_npz else "(no existing NPZ found)"
            raise KeyError(
                f"Missing required high-res constant forcing '{name}'. "
                f"Provide it in {constant_forcings_npz!s} or the bundle. "
                f"Expected flattened size={lat_hres.size}. "
                f"Resolved NPZ source={constant_source or 'none'}. "
                f"Tried: {tried}"
            )
        x_hres[0, 0, 0, :, name_to_idx_hres[name]] = torch.from_numpy(arr).to(device)

    # Compute dynamic forcings via earthkit to match anemoi-inference behavior.
    dynamic_features = [
        name
        for name in (
            "cos_julian_day",
            "sin_julian_day",
            "cos_local_time",
            "sin_local_time",
            "insolation",
        )
        if name in name_to_idx_hres
    ]
    if dynamic_features:
        source = UnstructuredGridFieldList.from_values(
            latitudes=np.asarray(lat_hres, dtype=np.float64),
            longitudes=np.asarray(lon_hres, dtype=np.float64),
        )
        forcings = ekd.from_source("forcings", source, date=[dt], param=dynamic_features)
        computed = {
            f.metadata("param"): np.asarray(f.to_numpy(flatten=True), dtype=np.float32)
            for f in forcings
        }
        for name in dynamic_features:
            if name not in computed:
                raise KeyError(f"Missing computed forcing from earthkit: {name}")
            x_hres[0, 0, 0, :, name_to_idx_hres[name]] = torch.from_numpy(
                computed[name]
            ).to(device)


def load_inputs_from_bundle_numpy(
    bundle_nc: str | Path | xr.Dataset,
    name_to_idx_lres: Mapping[str, int],
    name_to_idx_hres: Mapping[str, int],
    *,
    valid_time_override=None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle, should_close = _borrow_or_open_bundle_dataset(bundle_nc)
    try:
        n_lres = int(bundle.sizes["point_lres"])
        n_hres = int(bundle.sizes["point_hres"])

        x_lres = np.zeros((n_lres, len(name_to_idx_lres)), dtype=np.float32)
        x_hres = np.zeros((n_hres, len(name_to_idx_hres)), dtype=np.float32)

        levels_bundle = (
            set(int(v) for v in np.asarray(bundle["level"].values).tolist())
            if "level" in bundle
            else set()
        )

        # Read grid coordinates and valid time early so they are available for
        # computing dynamic temporal features on the LR side when absent from bundle.
        lat_lres = bundle["lat_lres"].values.astype(np.float32)
        lon_lres = bundle["lon_lres"].values.astype(np.float32)
        lat_hres = bundle["lat_hres"].values.astype(np.float32)
        lon_hres = bundle["lon_hres"].values.astype(np.float32)
        z = bundle["in_hres_z"].values.astype(np.float32) if "in_hres_z" in bundle else None
        lsm = bundle["in_hres_lsm"].values.astype(np.float32) if "in_hres_lsm" in bundle else None
        dt = parse_valid_time(valid_time_override, bundle.attrs.get("case_valid_time"))

        _DYNAMIC_TEMPORAL_FEATURES = frozenset(
            (
                "cos_julian_day", "sin_julian_day",
                "cos_local_time", "sin_local_time",
                "cos_solar_zenith_angle",
                "insolation",
            )
        )
        # Geometric constants computable directly from grid coordinates.
        _GEOMETRIC_CONSTANT_FEATURES = frozenset(
            ("cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude")
        )
        _lres_dynamic_cache: dict[str, np.ndarray] = {}

        # Load LR constant forcings (z, lsm, …) from npz if available for this grid size.
        _lres_npz_constants: dict[str, np.ndarray] = {}
        _lres_npz_candidates = (
            ([constant_forcings_npz] if constant_forcings_npz is not None else [])
            + list(FALLBACK_CONSTANT_FORCINGS_NPZ)
        )
        _lres_npz_constants, _, _ = _load_constant_forcings_for_size(
            _lres_npz_candidates, lat_lres.size
        )

        _zero_filled_vars: list[str] = []

        for name, idx in name_to_idx_lres.items():
            base, level = split_level_channel(name)
            if level is None:
                field_name = f"in_lres_{name}"
                if field_name not in bundle:
                    if name in _DYNAMIC_TEMPORAL_FEATURES:
                        # Compute via earthkit on the LR grid (same approach as HRES).
                        if not _lres_dynamic_cache:
                            import earthkit.data as ekd  # pylint: disable=import-outside-toplevel
                            from anemoi.transform.grids.unstructured import (
                                UnstructuredGridFieldList,
                            )  # pylint: disable=import-outside-toplevel

                            needed = [
                                n for n in _DYNAMIC_TEMPORAL_FEATURES if f"in_lres_{n}" not in bundle
                                and n in name_to_idx_lres
                            ]
                            source_lres = UnstructuredGridFieldList.from_values(
                                latitudes=np.asarray(lat_lres, dtype=np.float64),
                                longitudes=np.asarray(lon_lres, dtype=np.float64),
                            )
                            forcings_lres = ekd.from_source(
                                "forcings", source_lres, date=[dt], param=needed
                            )
                            _lres_dynamic_cache = {
                                f.metadata("param"): np.asarray(
                                    f.to_numpy(flatten=True), dtype=np.float32
                                )
                                for f in forcings_lres
                            }
                        if name not in _lres_dynamic_cache:
                            raise KeyError(
                                f"Missing bundle field: {field_name} (dynamic feature not returned by earthkit)"
                            )
                        raw = _lres_dynamic_cache[name]
                    elif name in _GEOMETRIC_CONSTANT_FEATURES:
                        # Trivially computable from grid coordinates.
                        lat_rad = np.deg2rad(lat_lres.astype(np.float64))
                        lon_rad = np.deg2rad(lon_lres.astype(np.float64))
                        _geo = {
                            "cos_latitude": np.cos(lat_rad),
                            "sin_latitude": np.sin(lat_rad),
                            "cos_longitude": np.cos(lon_rad),
                            "sin_longitude": np.sin(lon_rad),
                        }
                        raw = _geo[name].astype(np.float32)
                    elif name in _lres_npz_constants:
                        # Load from pre-built forcings npz (z, lsm, slor, …).
                        raw = _lres_npz_constants[name]
                    elif name in OPTIONAL_ZERO_LRES_SFC_VARS:
                        logger.warning(
                            "Zero-filling missing lres variable '%s' (index %d). "
                            "If the model was trained on real values for this variable, "
                            "zero-fill will be out-of-distribution and may corrupt predictions.",
                            name, idx,
                        )
                        _zero_filled_vars.append(name)
                        raw = np.zeros(n_lres, dtype=np.float32)
                    else:
                        raise KeyError(f"Missing bundle field: {field_name}")
                else:
                    candidate = bundle[field_name]
                    if "level" in candidate.dims:
                        if name in _lres_npz_constants:
                            raw = _lres_npz_constants[name]
                        else:
                            raise KeyError(
                                f"Missing bundle field: {field_name} (found pressure-level field where single-level input was expected)"
                            )
                    else:
                        raw = candidate.values.astype(np.float32)
            else:
                field_name = f"in_lres_{base}"
                if field_name not in bundle:
                    raise KeyError(f"Missing bundle field: {field_name}")
                if level not in levels_bundle:
                    raw = _interpolate_level_points(bundle[field_name], sorted(levels_bundle), int(level))
                else:
                    raw = bundle[field_name].sel(level=int(level)).values.astype(np.float32)
            x_lres[:, idx] = np.asarray(raw, dtype=np.float32).reshape(-1)

        import torch  # pylint: disable=import-outside-toplevel

        x_hres_tensor = torch.zeros((1, 1, 1, n_hres, len(name_to_idx_hres)), dtype=torch.float32)
        fill_hres_features(
            x_hres_tensor,
            name_to_idx_hres,
            lat_hres,
            lon_hres,
            dt,
            "cpu",
            z=z,
            lsm=lsm,
            constant_forcings_npz=constant_forcings_npz,
        )
        x_hres[:, :] = x_hres_tensor[0, 0, 0].numpy()
        for name, idx in name_to_idx_hres.items():
            if name in BUNDLE_IMPLICIT_HRES_FEATURES:
                continue
            raw = _extract_explicit_hres_input_channel(bundle, name)
            if raw is None:
                raise KeyError(
                    f"Missing explicit high-res input channel '{name}'. "
                    "Bundle inference can synthesize only forcing/static HRES inputs by default; "
                    "non-forcing HRES channels must be stored explicitly as in_hres_* fields in the bundle."
                )
            x_hres[:, idx] = np.asarray(raw, dtype=np.float32).reshape(-1)

        if _zero_filled_vars:
            logger.warning(
                "OOD RISK: %d lres variables were zero-filled: %s. "
                "Predictions may be unreliable if the model was trained with real values for these fields.",
                len(_zero_filled_vars), ", ".join(sorted(_zero_filled_vars)),
            )

        return x_lres, x_hres, lon_lres, lat_lres, lon_hres, lat_hres
    finally:
        if should_close:
            try:
                bundle.close()
            except Exception:
                pass


def fill_inputs_from_bundle(
    bundle_nc: str | Path,
    x_lres,
    x_hres,
    name_to_idx_lres: Mapping[str, int],
    name_to_idx_hres: Mapping[str, int],
    device,
    *,
    valid_time_override=None,
    constant_forcings_npz: str | Path | None = DEFAULT_CONSTANT_FORCINGS_NPZ,
) -> None:
    import torch  # pylint: disable=import-outside-toplevel
    x_lres_np, x_hres_np, _, _, _, _ = load_inputs_from_bundle_numpy(
        bundle_nc,
        name_to_idx_lres,
        name_to_idx_hres,
        valid_time_override=valid_time_override,
        constant_forcings_npz=constant_forcings_npz,
    )
    if int(x_lres_np.shape[0]) != int(x_lres.shape[3]):
        raise RuntimeError("LRES grid-size mismatch between bundle and model template")
    if int(x_hres_np.shape[0]) != int(x_hres.shape[3]):
        raise RuntimeError("HRES grid-size mismatch between bundle and model template")
    x_lres[0, 0, 0, :, :] = torch.from_numpy(x_lres_np).to(device)
    x_hres[0, 0, 0, :, :] = torch.from_numpy(x_hres_np).to(device)


def extract_target_from_bundle_dataset(
    bundle: xr.Dataset,
    weather_states: Sequence[str],
) -> tuple[np.ndarray | None, int]:
    """Return optional high-res target truth from bundle as [point_hres, weather_state].

    Supported variable naming in bundle:
    - point fields: `target_hres_<name>` (also `out_hres_<name>`, `y_hres_<name>`)
    - level fields: `target_hres_<base>` with level-like coord (`target_level`/`level`/`isobaricInhPa`)
      where weather state names use `<base>_<level>` (e.g. `t_850`).
    """
    n_points = int(bundle.sizes["point_hres"])
    y = np.full((n_points, len(weather_states)), np.nan, dtype=np.float32)
    found_channels = 0
    prefixes = ("target_hres_", "out_hres_", "y_hres_")

    for i, name in enumerate(weather_states):
        base, level = split_level_channel(name)
        candidates: list[str] = []
        for pref in prefixes:
            candidates.append(f"{pref}{name}")
            candidates.append(f"{pref}{base}")

        channel = None
        for var_name in candidates:
            if var_name not in bundle:
                continue
            da = bundle[var_name]
            if level is not None:
                selected = False
                for lev_name in ("target_level", "level", "isobaricInhPa"):
                    if lev_name in da.coords:
                        levels = np.asarray(da[lev_name].values).astype(int).tolist()
                        if int(level) not in levels:
                            break
                        da = da.sel({lev_name: int(level)})
                        selected = True
                        break
                if not selected and any(n in da.coords for n in ("target_level", "level", "isobaricInhPa")):
                    continue

            ok = True
            for dim in list(da.dims):
                if dim == "point_hres":
                    continue
                if da.sizes[dim] != 1:
                    ok = False
                    break
                da = da.isel({dim: 0})
            if not ok:
                continue

            vals = np.asarray(da.values, dtype=np.float32).reshape(-1)
            if vals.size != n_points:
                continue
            channel = vals
            break

        if channel is None:
            continue
        y[:, i] = channel
        found_channels += 1

    if found_channels == 0:
        return None, 0
    return y, found_channels


def extract_target_from_bundle(
    bundle_nc: str | Path | xr.Dataset,
    weather_states: Sequence[str],
) -> tuple[np.ndarray | None, int]:
    bundle, should_close = _borrow_or_open_bundle_dataset(bundle_nc)
    try:
        return extract_target_from_bundle_dataset(bundle, weather_states)
    finally:
        if should_close:
            try:
                bundle.close()
            except Exception:
                pass


def _to_1d_points(da: xr.DataArray) -> np.ndarray:
    da = da.copy()
    for dim in list(da.dims):
        if dim in ("values", "latitude", "longitude", "point_lres", "point_hres"):
            continue
        if da.sizes[dim] != 1:
            raise ValueError(f"Unexpected non-singleton dim {dim}={da.sizes[dim]}")
        da = da.isel({dim: 0})
    return np.asarray(da.values, dtype=np.float32).reshape(-1)


def _to_2d_level_points(da: xr.DataArray) -> np.ndarray:
    da = da.copy()
    keep = {"isobaricInhPa", "values", "level"}
    for d in list(da.dims):
        if d not in keep:
            if da.sizes[d] != 1:
                raise ValueError(
                    f"Unexpected non-singleton dim for 2D field: {d}={da.sizes[d]} in {da.name}"
                )
            da = da.isel({d: 0})
    if set(da.dims) == {"isobaricInhPa", "values"}:
        da = da.transpose("isobaricInhPa", "values")
    elif set(da.dims) == {"level", "values"}:
        da = da.transpose("level", "values")
    else:
        raise ValueError(f"Unexpected dims for PL field {da.name}: {da.dims}")
    return np.asarray(da.values, dtype=np.float32)


def _select_step(
    ds: xr.Dataset,
    step_hours: int | None,
    *,
    allow_any_single_step: bool = False,
) -> xr.Dataset:
    if step_hours is None:
        return ds
    if "step" not in ds.dims and "step" not in ds.coords:
        return ds
    target = np.timedelta64(int(step_hours), "h")
    try:
        return ds.sel(step=target)
    except Exception as exc:
        if allow_any_single_step:
            step_size = ds.sizes.get("step")
            if step_size == 1:
                return ds.isel(step=0)
        raise ValueError(f"Failed to select step={step_hours}h in dataset") from exc


def _select_member(
    ds: xr.Dataset, member: int | None, *, allow_missing: bool = False
) -> xr.Dataset:
    if member is None:
        return ds
    requested_member = int(member)

    def _select_named_member(dim_name: str) -> xr.Dataset | None:
        if dim_name not in ds.dims and dim_name not in ds.coords:
            return None

        coord = ds.coords.get(dim_name)
        if coord is not None:
            values = np.asarray(coord.values).reshape(-1)
            if values.size:
                matches = []
                for idx, raw_value in enumerate(values):
                    try:
                        if int(raw_value) == requested_member:
                            matches.append(idx)
                    except Exception:
                        continue
                if matches:
                    if dim_name in ds.dims:
                        return ds.isel({dim_name: int(matches[0])})
                    return ds
                if values.size == 1:
                    try:
                        only_value = int(values[0])
                    except Exception:
                        only_value = None
                    if allow_missing or (only_value == 0 and requested_member == 1):
                        if dim_name in ds.dims:
                            return ds.isel({dim_name: 0})
                        return ds

        if ds.sizes.get(dim_name, 0) == 1 and allow_missing:
            return ds.isel({dim_name: 0})
        return None

    for dim_name in ("number", "ensemble_member"):
        selected = _select_named_member(dim_name)
        if selected is not None:
            return selected
    return ds


_LRES_SFC_MAP = {
    "10u": "in_lres_10u",
    "10v": "in_lres_10v",
    "2d": "in_lres_2d",
    "2t": "in_lres_2t",
    "cp": "in_lres_cp",
    "hcc": "in_lres_hcc",
    "lcc": "in_lres_lcc",
    "mcc": "in_lres_mcc",
    "msl": "in_lres_msl",
    "skt": "in_lres_skt",
    "sp": "in_lres_sp",
    "ssrd": "in_lres_ssrd",
    "strd": "in_lres_strd",
    "tcc": "in_lres_tcc",
    "tcw": "in_lres_tcw",
    "tp": "in_lres_tp",
}

_TARGET_SFC_MAP = {
    "10u": "target_hres_10u",
    "10v": "target_hres_10v",
    "2d": "target_hres_2d",
    "2t": "target_hres_2t",
    "msl": "target_hres_msl",
    "skt": "target_hres_skt",
    "sp": "target_hres_sp",
    "tcw": "target_hres_tcw",
    "tp": "target_hres_tp",
}


def _build_lres_sfc_vars(
    ds_sfc,
    channels: Sequence[str],
    lat_lres: np.ndarray,
) -> dict:
    data_vars: dict = {}
    for src in channels:
        dst = _LRES_SFC_MAP.get(src)
        if dst is None:
            raise KeyError(f"Unsupported LRES SFC channel requested: {src}")
        cfgrib_name = SFC_TO_CFGRIB.get(src, src)
        if cfgrib_name not in ds_sfc:
            if src in OPTIONAL_ZERO_LRES_SFC_VARS:
                data_vars[dst] = ("point_lres", np.zeros(lat_lres.shape[0], dtype=np.float32))
                continue
            raise KeyError(f"Missing SFC variable in input: {src}")
        data_vars[dst] = ("point_lres", _to_1d_points(ds_sfc[cfgrib_name]))
    return data_vars


def _build_lres_pl_vars(
    ds_pl,
    channels: Sequence[str],
) -> dict:
    data_vars: dict = {}
    for v in channels:
        if ds_pl is None:
            raise KeyError(f"Requested PL channel {v} but no pressure-level dataset was loaded.")
        if v not in ds_pl:
            raise KeyError(f"Missing PL variable in input: {v}")
        data_vars[f"in_lres_{v}"] = (("level", "point_lres"), _to_2d_level_points(ds_pl[v]))
    return data_vars


def _build_target_sfc_vars(
    ds_target_sfc,
    channels: Sequence[str],
    lat_hres: np.ndarray,
) -> dict:
    data_vars: dict = {}
    for src in channels:
        dst = _TARGET_SFC_MAP.get(src)
        if dst is None:
            raise KeyError(f"Unsupported target SFC channel requested: {src}")
        cfgrib_name = SFC_TO_CFGRIB.get(src, src)
        if cfgrib_name not in ds_target_sfc:
            continue
        vals = _to_1d_points(ds_target_sfc[cfgrib_name])
        if vals.size != lat_hres.size:
            raise ValueError(
                f"Target SFC field {src} point count {vals.size} != point_hres {lat_hres.size}"
            )
        data_vars[dst] = ("point_hres", vals)
    return data_vars


def _build_target_pl_vars(
    ds_target_pl,
    channels: Sequence[str],
    lat_hres: np.ndarray,
) -> tuple[dict, np.ndarray | None]:
    data_vars: dict = {}
    level_coord_target = _get_pl_level_coord(ds_target_pl)
    target_level_coord = np.asarray(ds_target_pl[level_coord_target].values, dtype=np.int32).reshape(-1)
    for var in channels:
        if var not in ds_target_pl:
            continue
        vals = _to_2d_level_points(ds_target_pl[var])
        if vals.shape[1] != lat_hres.size:
            raise ValueError(
                f"Target PL field {var} point count {vals.shape[1]} != point_hres {lat_hres.size}"
            )
        data_vars[f"target_hres_{var}"] = (("target_level", "point_hres"), vals)
    return data_vars, target_level_coord


def _write_bundle_metadata(
    bundle: xr.Dataset,
    *,
    lres_sfc_grib: str | Path,
    lres_sfc_extra_gribs: Sequence[str | Path],
    lres_pl_grib: str | Path,
    hres_grib: str | Path,
    hres_static_grib: str | Path | None,
    target_sfc_grib: str | Path | None,
    target_sfc_extra_gribs: Sequence[str | Path],
    target_pl_grib: str | Path | None,
    resolved_channels: dict[str, Sequence[str]],
    ds_sfc,
    step_hours: int | None,
    member: int | None,
    has_target: bool,
    require_target_fields: bool,
) -> None:
    valid_time = (
        str(np.asarray(ds_sfc["valid_time"].values).squeeze())
        if "valid_time" in ds_sfc
        else "unknown"
    )
    bundle.attrs["case_valid_time"] = valid_time
    bundle.attrs["source_lres_sfc"] = str(lres_sfc_grib)
    if lres_sfc_extra_gribs:
        bundle.attrs["source_lres_sfc_extra"] = ",".join(str(path) for path in lres_sfc_extra_gribs)
    bundle.attrs["source_lres_pl"] = str(lres_pl_grib)
    resolved_hres = hres_static_grib or hres_grib
    bundle.attrs["source_hres"] = str(resolved_hres)
    if hres_static_grib:
        bundle.attrs["source_hres_static_override"] = str(hres_static_grib)
    if target_sfc_grib:
        bundle.attrs["source_target_sfc"] = str(target_sfc_grib)
    if target_sfc_extra_gribs:
        bundle.attrs["source_target_sfc_extra"] = ",".join(str(path) for path in target_sfc_extra_gribs)
    if target_pl_grib:
        bundle.attrs["source_target_pl"] = str(target_pl_grib)
    bundle.attrs["selected_lres_sfc_channels"] = ",".join(resolved_channels["lres_sfc"])
    bundle.attrs["selected_lres_pl_channels"] = ",".join(resolved_channels["lres_pl"])
    bundle.attrs["selected_target_sfc_channels"] = ",".join(resolved_channels["target_sfc"])
    bundle.attrs["selected_target_pl_channels"] = ",".join(resolved_channels["target_pl"])
    bundle.attrs["has_target_hres_fields"] = "yes" if has_target else "no"
    if has_target:
        bundle.attrs["description"] = (
            "Combined low-res + high-res feature inputs for local inference. "
            "Includes target_hres_* fields for truth-aware evaluation."
        )
    else:
        bundle.attrs["description"] = (
            "Combined low-res + high-res feature inputs for local inference. "
            "Created without target_hres_* fields because --allow-missing-target-unsafe was used. "
            "Prediction-only and non-canonical for truth-aware evaluation."
        )
        bundle.attrs["missing_target_policy"] = "bundle_without_target_hres_due_to_allow_missing_target_unsafe"
    if step_hours is not None:
        bundle.attrs["step_hours"] = int(step_hours)
    if member is not None:
        bundle.attrs["member"] = int(member)


def build_input_bundle_from_grib(
    *,
    lres_sfc_grib: str | Path,
    lres_sfc_extra_gribs: Sequence[str | Path] | None = None,
    lres_pl_grib: str | Path,
    hres_grib: str | Path,
    hres_static_grib: str | Path | None = None,
    out_nc: str | Path,
    step_hours: int | None = None,
    member: int | None = None,
    out_zarr: str | Path | None = None,
    target_sfc_grib: str | Path | None = None,
    target_sfc_extra_gribs: Sequence[str | Path] | None = None,
    target_pl_grib: str | Path | None = None,
    require_target_fields: bool = True,
    lres_sfc_channels: Sequence[str] | None = None,
    lres_pl_channels: Sequence[str] | None = None,
    target_sfc_channels: Sequence[str] | None = None,
    target_pl_channels: Sequence[str] | None = None,
) -> Path:
    resolved_lres_sfc_channels = _normalize_channel_subset(
        lres_sfc_channels,
        default=DEFAULT_LRES_SFC_CHANNELS,
    )
    resolved_lres_pl_channels = _normalize_channel_subset(
        lres_pl_channels,
        default=DEFAULT_LRES_PL_CHANNELS,
    )
    resolved_target_sfc_channels = _normalize_channel_subset(
        target_sfc_channels,
        default=DEFAULT_TARGET_SFC_CHANNELS,
    )
    resolved_target_pl_channels = _normalize_channel_subset(
        target_pl_channels,
        default=DEFAULT_TARGET_PL_CHANNELS,
    )
    resolved_lres_sfc_extra_gribs = _normalize_optional_grib_list(lres_sfc_extra_gribs)
    resolved_target_sfc_extra_gribs = _normalize_optional_grib_list(target_sfc_extra_gribs)

    # DestinE low-resolution inputs may be packaged as a single mixed-level GRIB.
    # Reopen the same file with explicit cfgrib level filters so bundle creation
    # does not depend on pre-splitting surface and pressure-level inputs.
    ds_sfc = _open_selected_surface_dataset(
        lres_sfc_grib,
        resolved_lres_sfc_extra_gribs,
        step_hours=step_hours,
        member=member,
        filter_by_keys={"typeOfLevel": "surface"},
    )
    ds_pl = None
    if resolved_lres_pl_channels:
        ds_pl = _open_cfgrib_dataset(
            lres_pl_grib,
            filter_by_keys={"typeOfLevel": "isobaricInhPa"},
        )
    resolved_hres_static_grib = hres_static_grib or hres_grib
    ds_hres = _open_cfgrib_dataset(resolved_hres_static_grib)

    if ds_pl is not None:
        ds_pl = _select_step(ds_pl, step_hours)
    ds_hres = _select_step(
        ds_hres,
        step_hours,
        allow_any_single_step=bool(hres_static_grib),
    )

    if ds_pl is not None:
        ds_pl = _select_member(ds_pl, member)
    ds_hres = _select_member(ds_hres, member, allow_missing=True)

    lat_lres = _to_1d_points(ds_sfc["latitude"])
    lon_lres = _to_1d_points(ds_sfc["longitude"])
    lat_hres = _to_1d_points(ds_hres["latitude"])
    lon_hres = _to_1d_points(ds_hres["longitude"])

    coords = {
        "point_lres": np.arange(lat_lres.shape[0], dtype=np.int32),
        "point_hres": np.arange(lat_hres.shape[0], dtype=np.int32),
        "lat_lres": ("point_lres", lat_lres),
        "lon_lres": ("point_lres", lon_lres),
        "lat_hres": ("point_hres", lat_hres),
        "lon_hres": ("point_hres", lon_hres),
    }
    if ds_pl is not None:
        level_coord = _get_pl_level_coord(ds_pl)
        levels = np.asarray(ds_pl[level_coord].values, dtype=np.int32).reshape(-1)
        coords["level"] = levels

    data_vars: dict = {}
    data_vars.update(_build_lres_sfc_vars(ds_sfc, resolved_lres_sfc_channels, lat_lres))
    data_vars.update(_build_lres_pl_vars(ds_pl, resolved_lres_pl_channels))

    hres_map = {"z": "in_hres_z", "lsm": "in_hres_lsm"}
    for src, dst in hres_map.items():
        if src not in ds_hres:
            continue
        data_vars[dst] = ("point_hres", _to_1d_points(ds_hres[src]))

    auto_sfc, auto_pl = (None, None)
    if require_target_fields and target_sfc_grib is None and target_pl_grib is None:
        auto_sfc, auto_pl = _infer_target_gribs_from_hres(hres_grib)
    target_sfc_grib = target_sfc_grib or auto_sfc
    target_pl_grib = target_pl_grib or auto_pl

    target_level_coord: np.ndarray | None = None
    if target_sfc_grib:
        ds_target_sfc = _open_selected_surface_dataset(
            target_sfc_grib,
            resolved_target_sfc_extra_gribs,
            step_hours=step_hours,
            member=member,
            allow_missing_member=True,
        )
        data_vars.update(_build_target_sfc_vars(ds_target_sfc, resolved_target_sfc_channels, lat_hres))

    if target_pl_grib and resolved_target_pl_channels:
        ds_target_pl = _open_cfgrib_dataset(target_pl_grib)
        ds_target_pl = _select_step(ds_target_pl, step_hours)
        ds_target_pl = _select_member(ds_target_pl, member, allow_missing=True)
        pl_vars, target_level_coord = _build_target_pl_vars(ds_target_pl, resolved_target_pl_channels, lat_hres)
        data_vars.update(pl_vars)

    has_target = any(name.startswith("target_hres_") for name in data_vars)
    if require_target_fields and not has_target:
        raise ValueError(
            "No target_hres_* fields were added to bundle. "
            "Provide --target-sfc-grib/--target-pl-grib or place matching target *_y.grib files near hres input."
        )

    bundle = xr.Dataset(data_vars=data_vars, coords=coords)
    if target_level_coord is not None:
        bundle = bundle.assign_coords(target_level=target_level_coord)

    _write_bundle_metadata(
        bundle,
        lres_sfc_grib=lres_sfc_grib,
        lres_sfc_extra_gribs=resolved_lres_sfc_extra_gribs,
        lres_pl_grib=lres_pl_grib,
        hres_grib=hres_grib,
        hres_static_grib=hres_static_grib,
        target_sfc_grib=target_sfc_grib,
        target_sfc_extra_gribs=resolved_target_sfc_extra_gribs,
        target_pl_grib=target_pl_grib,
        resolved_channels={
            "lres_sfc": resolved_lres_sfc_channels,
            "lres_pl": resolved_lres_pl_channels,
            "target_sfc": resolved_target_sfc_channels,
            "target_pl": resolved_target_pl_channels,
        },
        ds_sfc=ds_sfc,
        step_hours=step_hours,
        member=member,
        has_target=has_target,
        require_target_fields=require_target_fields,
    )

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle.to_netcdf(out_nc)
    except Exception:
        bundle.to_netcdf(out_nc, engine="scipy")
    if out_zarr:
        out_zarr = Path(out_zarr)
        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        bundle.to_zarr(out_zarr, mode="w", consolidated=True)
    return out_nc


def _default_bundle_name(lres_sfc_grib: str | Path) -> str:
    name = Path(lres_sfc_grib).name
    if name.endswith("_sfc.grib"):
        name = name[: -len("_sfc.grib")]
    elif name.endswith(".grib"):
        name = name[: -len(".grib")]
    return f"{name}_input_bundle.nc"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build input bundle NetCDF from GRIB.")
    parser.add_argument("--lres-sfc-grib", required=True)
    parser.add_argument(
        "--lres-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra low-resolution surface GRIB to merge before bundle creation.",
    )
    parser.add_argument("--lres-pl-grib", required=True)
    parser.add_argument("--hres-grib", required=True)
    parser.add_argument(
        "--hres-static-grib",
        default="",
        help=(
            "Optional GRIB used only for HRES static/input fields such as z/lsm. "
            "Defaults to --hres-grib when omitted."
        ),
    )
    parser.add_argument(
        "--target-sfc-grib",
        default="",
        help="Optional high-res surface GRIB containing target/truth fields to store in bundle.",
    )
    parser.add_argument(
        "--target-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra target surface GRIB to merge before bundle creation.",
    )
    parser.add_argument(
        "--target-pl-grib",
        default="",
        help="Optional high-res pressure-level GRIB containing target/truth fields to store in bundle.",
    )
    parser.add_argument(
        "--lres-sfc-channels",
        default="",
        help="Optional CSV override for low-resolution surface bundle channels.",
    )
    parser.add_argument(
        "--lres-pl-channels",
        default="",
        help="Optional CSV override for low-resolution pressure-level bundle channels.",
    )
    parser.add_argument(
        "--target-sfc-channels",
        default="",
        help="Optional CSV override for target high-resolution surface bundle channels.",
    )
    parser.add_argument(
        "--target-pl-channels",
        default="",
        help="Optional CSV override for target high-resolution pressure-level bundle channels.",
    )
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Deprecated alias. Use --allow-missing-target-unsafe instead.",
    )
    parser.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow creating bundle without target_hres_* fields. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--out-zarr",
        default="",
        help="Optional Zarr output for faster chunked reads.",
    )
    parser.add_argument(
        "--out-root",
        default="/home/ecm5702/hpcperm/data/input_data",
        help="Base output folder for input data.",
    )
    parser.add_argument("--grid", default="", help="Resolution tag (e.g. o320, o96).")
    parser.add_argument(
        "--step-hours",
        type=int,
        default=None,
        help="Select a single step (hours) from multi-step GRIBs.",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=None,
        help="Select a single member if GRIBs contain an ensemble dimension.",
    )
    args = parser.parse_args()
    if args.allow_missing_target:
        raise SystemExit(
            "--allow-missing-target is deprecated. "
            "Use --allow-missing-target-unsafe for an explicit prediction-only escape hatch."
        )

    out_path = args.out
    if not out_path:
        grid = args.grid or "grid"
        out_dir = Path(args.out_root) / grid
        out_path = out_dir / _default_bundle_name(args.lres_sfc_grib)

    out = build_input_bundle_from_grib(
        lres_sfc_grib=args.lres_sfc_grib,
        lres_sfc_extra_gribs=args.lres_sfc_extra_grib,
        lres_pl_grib=args.lres_pl_grib,
        hres_grib=args.hres_grib,
        hres_static_grib=args.hres_static_grib or None,
        out_nc=out_path,
        step_hours=args.step_hours,
        member=args.member,
        out_zarr=args.out_zarr or None,
        target_sfc_grib=args.target_sfc_grib or None,
        target_sfc_extra_gribs=args.target_sfc_extra_grib,
        target_pl_grib=args.target_pl_grib or None,
        require_target_fields=not args.allow_missing_target_unsafe,
        lres_sfc_channels=_parse_channel_subset_csv(args.lres_sfc_channels),
        lres_pl_channels=_parse_channel_subset_csv(args.lres_pl_channels),
        target_sfc_channels=_parse_channel_subset_csv(args.target_sfc_channels),
        target_pl_channels=_parse_channel_subset_csv(args.target_pl_channels),
    )
    print(f"Saved bundle: {out}")


if __name__ == "__main__":
    main()
