"""Convert raw MARS GRIB (from fetch_mars_grib.py) into the MeteoSwiss IFS-HRES
GRIB archive layout (see /capstor/store/mch/msopr/osm/IFS-HRES/IFS-HRESyy/).

Output layout matches that archive's convention (same one compare_zarr_to_grib.py
reads): one directory per base time, one file per lead time.

    {output_dir}/{YYMMDDHH}/efsf{step//24:02d}{step%24:02d}0000

Only a subset of the archive's variables is produced -- 2t, 10u, 10v, tp
(surface) and t, u, v, z at 500/850 hPa -- everything else the real archive
carries (hybrid-level fields, clouds, radiation, snow, ...) is simply absent.
The horizontal grid is also not reprojected onto the archive's 0.1deg rotated
grid; output stays on a regular (full) Gaussian grid, cropped to the
geographic area that rotated grid covers.

u/v wind is not in the MARS input -- fetch_mars_grib.py deliberately retrieves
vorticity/divergence (138/155) instead of u/v (131/132) to avoid the MARS-side
wind derivation cost. Recovering u/v from vo/d is a global spherical-harmonic
(Helmholtz) inversion, done here via CDO:

    cdo dv2uv -gp2sp -setgridtype,regular   (reduced Gaussian vo/d -> full Gaussian u/v)

Requires the `cdo` binary on PATH (not just the `cdo` Python package, which is
only a thin wrapper around it).

Usage:
    python src/hirad/input_data/ifs-hres/mars_to_ifs_hres.py \\
        --input-dir /capstor/scratch/cscs/mmcgloho/mars/ \\
        --output-dir /capstor/scratch/cscs/mmcgloho/mars/ifs_hres_format/
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import eccodes as ec

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Geographic bounding box of the MeteoSwiss COSMO rotated-pole archive grid
# (pole at lat=-43, lon=10; domain Ni=264,Nj=223 @ 0.1deg), with a small margin.
DEFAULT_LON_MIN = -23.0
DEFAULT_LON_MAX = 25.0
DEFAULT_LAT_MIN = 33.0
DEFAULT_LAT_MAX = 59.0

LEVELS = (500, 850)
SFC_SHORTNAMES = {"2t", "10u", "10v", "tp"}
PL_TZ_SHORTNAMES = {"t", "z"}
PL_UV_SHORTNAMES = {"u", "v"}


def basetime_key(date: int, time: int) -> str:
    """dataDate (YYYYMMDD) + dataTime (HMM/HHMM) -> 'YYMMDDHH'."""
    hour = time // 100
    return f"{date % 10000:04d}{hour:02d}"


def grib_filename(step_hours: int) -> str:
    day, hour = divmod(step_hours, 24)
    return f"efsf{day:02d}{hour:02d}0000"


def split_by_basetime(source_globs: list[str], tmp_dir: Path, suffix: str) -> set[str]:
    """Stream every message in the matching source files into one temp file per
    basetime (dataDate+dataTime), named `{tmp_dir}/{basetime}_{suffix}.grib`.

    Streams message-by-message so a month of global O1280 data never sits in
    memory all at once.
    """
    handles: dict[str, object] = {}
    basetimes: set[str] = set()
    n_files = 0
    for pattern in source_globs:
        for path in sorted(Path().glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("/").glob(pattern[1:])):
            n_files += 1
            LOG.info("Splitting %s by basetime...", path)
            with open(path, "rb") as f:
                while True:
                    gid = ec.codes_grib_new_from_file(f)
                    if gid is None:
                        break
                    date = ec.codes_get(gid, "dataDate")
                    time = ec.codes_get(gid, "dataTime")
                    key = basetime_key(date, time)
                    basetimes.add(key)
                    if key not in handles:
                        handles[key] = open(tmp_dir / f"{key}_{suffix}.grib", "ab")
                    ec.codes_write(gid, handles[key])
                    ec.codes_release(gid)
    for h in handles.values():
        h.close()
    if n_files == 0:
        LOG.warning("No source files matched for suffix=%s (globs=%s)", suffix, source_globs)
    return basetimes


def run_cdo(args: list[str]) -> None:
    LOG.info("cdo %s", " ".join(args))
    subprocess.run(["cdo", "-s", *args], check=True, capture_output=True, text=True)


def process_basetime(
    key: str,
    tmp_dir: Path,
    output_dir: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> None:
    pl_in = tmp_dir / f"{key}_pl.grib"
    sfc_in = tmp_dir / f"{key}_sfc.grib"

    box = f"{lon_min},{lon_max},{lat_min},{lat_max}"
    tz_out = tmp_dir / f"{key}_tz.grib"
    uv_out = tmp_dir / f"{key}_uv.grib"
    sfc_out = tmp_dir / f"{key}_sfc_cropped.grib"

    if pl_in.exists():
        run_cdo([f"-sellonlatbox,{box}", "-setgridtype,regular", "-selname,z,t", str(pl_in), str(tz_out)])
        # vo/d -> u/v: select first (cheap), regrid reduced->full Gaussian, forward
        # SH transform, invert vorticity/divergence to streamfunction/velocity
        # potential, derive u/v, then crop.
        run_cdo(
            [
                f"-sellonlatbox,{box}",
                "-dv2uv",
                "-gp2sp",
                "-setgridtype,regular",
                "-selname,vo,d",
                str(pl_in),
                str(uv_out),
            ]
        )
    else:
        LOG.warning("No pressure-level data for basetime %s, skipping t/z/u/v", key)
        tz_out = uv_out = None

    if sfc_in.exists():
        run_cdo([f"-sellonlatbox,{box}", "-setgridtype,regular", str(sfc_in), str(sfc_out)])
    else:
        LOG.warning("No surface data for basetime %s, skipping 2t/10u/10v/tp", key)
        sfc_out = None

    out_base = output_dir / key
    out_base.mkdir(parents=True, exist_ok=True)
    n_written = _assemble_steps(sfc_out, tz_out, uv_out, out_base)
    LOG.info("Basetime %s: wrote %d step files to %s", key, n_written, out_base)


def _assemble_steps(sfc_path, tz_path, uv_path, out_base: Path) -> int:
    """Stream the cropped per-field files and append each wanted message to its
    step's output file (creating it on first write)."""
    steps_seen: set[int] = set()

    def copy_messages(path, wanted_shortnames, wanted_levels=None):
        if path is None:
            return
        with open(path, "rb") as f:
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                short_name = ec.codes_get(gid, "shortName")
                if short_name not in wanted_shortnames:
                    ec.codes_release(gid)
                    continue
                if wanted_levels is not None and ec.codes_get(gid, "level") not in wanted_levels:
                    ec.codes_release(gid)
                    continue
                step = ec.codes_get(gid, "step")
                steps_seen.add(step)
                out_path = out_base / grib_filename(step)
                with open(out_path, "ab") as out_f:
                    ec.codes_write(gid, out_f)
                ec.codes_release(gid)

    copy_messages(sfc_path, SFC_SHORTNAMES)
    copy_messages(tz_path, PL_TZ_SHORTNAMES, LEVELS)
    copy_messages(uv_path, PL_UV_SHORTNAMES, LEVELS)
    return len(steps_seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with fetch_mars_grib.py's output.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Root of the IFS-HRES-style output archive.")
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the intermediate per-basetime GRIB files.")
    args = parser.parse_args()

    if shutil.which("cdo") is None:
        raise SystemExit("cdo binary not found on PATH -- required for the vo/d -> u/v inversion (dv2uv).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="mars_to_ifs_hres_", dir=args.output_dir) if not args.keep_tmp else None
    tmp_dir = Path(tmp_ctx.name) if tmp_ctx else args.output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        sfc_globs = [str(args.input_dir / "ifs_operational_surface_*.grib")]
        pl_globs = [
            str(args.input_dir / "ifs_operational_pl_av_short_*.grib"),
            str(args.input_dir / "ifs_operational_pl_av_long_*.grib"),
        ]

        LOG.info("Splitting surface files by basetime...")
        sfc_basetimes = split_by_basetime(sfc_globs, tmp_dir, "sfc")
        LOG.info("Splitting pressure-level files by basetime...")
        pl_basetimes = split_by_basetime(pl_globs, tmp_dir, "pl")

        basetimes = sorted(sfc_basetimes | pl_basetimes)
        LOG.info("Found %d basetimes: %s ... %s", len(basetimes), basetimes[:1], basetimes[-1:])

        for key in basetimes:
            process_basetime(key, tmp_dir, args.output_dir, args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()
        LOG.info("Done. Output written to %s", args.output_dir)


if __name__ == "__main__":
    main()
