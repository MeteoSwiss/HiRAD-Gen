"""Spot-check an anemoi trajectories zarr against the MeteoSwiss IFS-HRES GRIB
archive at /capstor/store/mch/msopr/osm/IFS-HRES/IFS-HRES25/.

Sanity check, not a rigorous validation: compares single-nearest-grid-point
values at a handful of locations, since the two products sit on different
grids (zarr: native O1280 octahedral reduced Gaussian; GRIB archive: a
regular 0.1deg rotated-pole grid). Small, *stable* offsets across steps at a
given point are expected -- they mean the two grids' nearest points to that
location aren't quite the same spot, more so near steep terrain. Growing or
erratic offsets would indicate an actual problem.

`tp` is deaccumulated from the GRIB's accumulated-since-forecast-start
convention (same as raw MARS) to make it comparable to the zarr's already
hourly-deaccumulated `tp`. `tcw` has no equivalent in the GRIB archive (it
only carries model-level `q`, not the vertically integrated total) and is
skipped.

Usage:
    python src/hirad/input_data/compare_zarr_to_grib.py \\
        --zarr /capstor/scratch/cscs/mmcgloho/anemoitest/leadtime1h.zarr \\
        --date 2025-01-01 \\
        --steps 1 3 6
"""

import argparse
import datetime
from pathlib import Path

import eccodes as ec
import numpy as np
import zarr

GRIB_ROOT = Path("/capstor/store/mch/msopr/osm/IFS-HRES/IFS-HRES25")

# zarr variable name -> (GRIB shortName, typeOfLevel, level or None)
VARIABLE_MAP = {
    "2t": ("2t", "surface", None),
    "10u": ("10u", "surface", None),
    "10v": ("10v", "surface", None),
    "tp": ("tp", "surface", None),  # accumulated-from-start in GRIB; deaccumulated below
    "t_500": ("t", "isobaricInhPa", 500),
    "t_850": ("t", "isobaricInhPa", 850),
    "z_500": ("z", "isobaricInhPa", 500),
    "z_850": ("z", "isobaricInhPa", 850),
    "u_500": ("u", "isobaricInhPa", 500),
    "u_850": ("u", "isobaricInhPa", 850),
    "v_500": ("v", "isobaricInhPa", 500),
    "v_850": ("v", "isobaricInhPa", 850),
}

DEFAULT_POINTS = {
    "Zurich": (47.37, 8.55),
    "Milan": (45.46, 9.19),
    "Munich": (48.14, 11.58),
    "domain edge (N)": (50.0, 8.0),
}


def grib_dir(base_date: datetime.date, cycle_hour: int = 0) -> Path:
    return GRIB_ROOT / f"{base_date:%y%m%d}{cycle_hour:02d}"


def grib_filename(step_hours: int) -> str:
    day, hour = divmod(step_hours, 24)
    return f"efsf{day:02d}{hour:02d}0000"


def read_grib_step(path: Path, wanted: set) -> dict:
    """Read one GRIB file, return {(shortName, typeOfLevel, level): nearest-fn}.

    `nearest-fn` is a closure over the opened codes id that finds the nearest
    grid point value to a given (lat, lon) -- kept open so callers can query
    several points without re-parsing the file.
    """
    found = {}
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                key = (ec.codes_get(gid, "shortName"), ec.codes_get(gid, "typeOfLevel"), ec.codes_get(gid, "level"))
            except ec.KeyValueNotFoundError:
                # A handful of local/derived params (e.g. paramId 260510) don't
                # expose a normal typeOfLevel/level -- none of ours, skip them.
                ec.codes_release(gid)
                continue
            key_no_level = (key[0], key[1], None)
            if key in wanted or key_no_level in wanted:
                found[key if key in wanted else key_no_level] = gid
            else:
                ec.codes_release(gid)
    return found


def grib_nearest_value(gid, lat: float, lon: float) -> float:
    return ec.codes_grib_find_nearest(gid, lat, lon)[0].value


def nearest_zarr_cell(zlat: np.ndarray, zlon_signed: np.ndarray, lat: float, lon: float) -> int:
    d2 = (zlat - lat) ** 2 + (zlon_signed - lon) ** 2
    return int(np.argmin(d2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", required=True, type=Path, help="Path to the anemoi trajectories zarr.")
    parser.add_argument("--grib-root", type=Path, default=GRIB_ROOT, help="MCH IFS-HRES GRIB archive root.")
    parser.add_argument("--date", required=True, type=datetime.date.fromisoformat, help="Base date, e.g. 2025-01-01.")
    parser.add_argument("--cycle-hour", type=int, default=0, help="Forecast cycle hour (the zarr only has 00Z).")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 3, 6], help="Lead times (hours) to compare.")
    parser.add_argument(
        "--variables", nargs="+", default=list(VARIABLE_MAP), choices=list(VARIABLE_MAP), help="zarr variable names."
    )
    parser.add_argument(
        "--point",
        action="append",
        dest="points",
        metavar="NAME:LAT:LON",
        help="Add a point to compare, e.g. --point Bern:46.95:7.45. Repeatable. Defaults to a small preset list.",
    )
    args = parser.parse_args()

    points = DEFAULT_POINTS.copy()
    if args.points:
        points = {}
        for spec in args.points:
            name, lat, lon = spec.split(":")
            points[name] = (float(lat), float(lon))

    grib_root_dir = grib_dir(args.date, args.cycle_hour)
    if not grib_root_dir.is_dir():
        raise SystemExit(f"No GRIB archive for {args.date} {args.cycle_hour:02d}Z: {grib_root_dir}")

    z = zarr.open(str(args.zarr), mode="r")
    zlat = z["latitudes"][:]
    zlon_signed = ((z["longitudes"][:] + 180) % 360) - 180
    zvariables = list(z.attrs["variables"])
    base_dates = z["base_dates"][:].astype("datetime64[s]")

    target = np.datetime64(datetime.datetime.combine(args.date, datetime.time(args.cycle_hour)))
    matches = np.where(base_dates == target)[0]
    if not len(matches):
        raise SystemExit(f"{target} not found in {args.zarr}'s base_dates.")
    date_idx = int(matches[0])

    wanted_keys = {VARIABLE_MAP[v] for v in args.variables}
    if "tp" in args.variables:
        wanted_keys.add(VARIABLE_MAP["tp"])  # needed at every step, including step-1, for deaccumulation

    # One GRIB read per step (covers all variables + all points for that step).
    grib_by_step = {}
    for step in sorted(set(args.steps) | ({s - 1 for s in args.steps if s > 1} if "tp" in args.variables else set())):
        if step == 0:
            continue
        path = grib_root_dir / grib_filename(step)
        grib_by_step[step] = read_grib_step(path, wanted_keys)

    for point_name, (lat, lon) in points.items():
        cell = nearest_zarr_cell(zlat, zlon_signed, lat, lon)
        print(f"\n--- {point_name} (requested {lat},{lon}; zarr nearest cell {zlat[cell]:.2f},{zlon_signed[cell]:.2f}) ---")

        for step in args.steps:
            parts = []
            for var in args.variables:
                key = VARIABLE_MAP[var]
                zval = z["data"][date_idx, zvariables.index(var), 0, step - 1, cell]

                gid = grib_by_step[step].get(key)
                if gid is None:
                    parts.append(f"{var}: <missing in GRIB>")
                    continue
                gval = grib_nearest_value(gid, lat, lon)

                if var == "tp":
                    gprev = 0.0 if step == 1 else grib_nearest_value(grib_by_step[step - 1][key], lat, lon)
                    gval = gval - gprev
                    parts.append(f"{var}: zarr={zval * 1000:.3f}mm grib={gval * 1000:.3f}mm")
                else:
                    parts.append(f"{var}: zarr={zval:.2f} grib={gval:.2f} (diff={zval - gval:+.2f})")

            print(f"  step={step:>2}h  " + "   ".join(parts))


if __name__ == "__main__":
    main()
