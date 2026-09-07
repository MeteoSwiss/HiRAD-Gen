"""Precompute IFS-HRES forecast input channels into per-(init, step) GRIB files.

Each output GRIB holds the 13 downscaling input channels on the native rotated
IFS-HRES grid for one (reference time, forecast step) pair:

    2t, 10u, 10v, tcw, t_850, z_850, u_850, v_850, t_500, z_500, u_500, v_500, tp

- The 8 pressure-level fields (t/z/u/v @ 850,500) and 3 surface fields (2t/10u/10v)
  are passed through unchanged; `z` stays geopotential (m^2/s^2), no conversion.
- `tcw` is derived from specific humidity `q` on hybrid levels and surface pressure
  `sp`:  tcw = sum(q * dp) / g,  with p_half = A + B * sp from the GRIB `pv` array.
- `tp` is a 1h accumulation, tp(step) - tp(step-1) (tp is accumulated-from-init, so
  step 0 is all-zero and is never emitted as a sample).

The two derived fields are encoded with low-level eccodes: earthkit-data 0.20.0's
Field.clone(values=..., paramId=...) leaks metadata overrides across fields.

Only the 00Z and 12Z IFS-HRES runs carry the full 3D fields (pressure levels +
hybrid q); the 06Z/18Z runs are reduced surface-only and must not be used as
inits here. Run (one init cycle -> all its leads) inside the container via SLURM;
see ifs_hres_precompute.sh.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List

import eccodes as ec
import earthkit.data as ekd
import numpy as np
from earthkit.meteo import constants

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ifs_hres_precompute")

PRESSURE_LEVELS = [850, 500]
PRESSURE_VARS = ["t", "z", "u", "v"]
SURFACE_VARS = ["2t", "10u", "10v"]
CHANNEL_ORDER = (
    SURFACE_VARS
    + ["tcw"]
    + [f"{v}_{lv}" for lv in PRESSURE_LEVELS for v in PRESSURE_VARS]
    + ["tp"]
)

PARAM_SP = 134   # surface pressure, template for tcw
PARAM_TCW = 136  # total column water
PARAM_TP = 228   # total precipitation


def ddhh(hours: int) -> str:
    """Forecast step in the IFS-HRES DDHH file-suffix encoding (e.g. 90 -> '0318')."""
    return f"{hours // 24:02d}{hours % 24:02d}"


def step_file(init_dir: str, hours: int) -> str:
    return os.path.join(init_dir, f"efsf{ddhh(hours)}0000")


def derive_tcw(ds) -> np.ndarray:
    """Total column water from hybrid-level q and surface pressure."""
    qv = ds.sel(shortName="q", typeOfLevel="hybrid").order_by(level="ascending")
    ps = ds.sel(shortName="sp", typeOfLevel="surface")[0].to_numpy()
    levels = [int(l) for l in qv.metadata("level")]
    pv = np.asarray(qv[0].metadata("pv"), dtype=float)
    nhalf = pv.size // 2
    a, b = pv[:nhalf], pv[nhalf:]
    shp = (-1,) + (1,) * ps.ndim
    p_half = (a.reshape(shp) + b.reshape(shp) * ps[None, ...])[levels[0] - 1: levels[-1] + 1]
    return (qv.to_numpy() * np.diff(p_half, axis=0)).sum(axis=0) / constants.g


def compute_channels(init_dir: str, hours: int):
    """Return (direct_fields, values) for one (init, step).

    direct_fields is an earthkit FieldList of the 11 pass-through channels;
    values maps every channel name (incl. derived tcw/tp) to its ndarray.
    """
    ds = ekd.from_source("file", step_file(init_dir, hours)).to_fieldlist()

    direct = ds.sel(shortName=SURFACE_VARS, typeOfLevel="surface") + \
        ds.sel(shortName=PRESSURE_VARS, typeOfLevel="isobaricInhPa", level=PRESSURE_LEVELS)

    values = {}
    for f in ds.sel(shortName=SURFACE_VARS, typeOfLevel="surface"):
        values[f.metadata("shortName")] = f.to_numpy()
    for f in ds.sel(shortName=PRESSURE_VARS, typeOfLevel="isobaricInhPa", level=PRESSURE_LEVELS):
        values[f"{f.metadata('shortName')}_{int(f.metadata('level'))}"] = f.to_numpy()

    values["tcw"] = derive_tcw(ds)

    tp_lead = ds.sel(shortName="tp", typeOfLevel="surface")[0].to_numpy()
    tp_prev = ekd.from_source("file", step_file(init_dir, hours - 1)).to_fieldlist() \
        .sel(shortName="tp", typeOfLevel="surface")[0].to_numpy()
    values["tp"] = tp_lead - tp_prev

    ref = values[SURFACE_VARS[0]]
    for name, arr in values.items():
        if arr.shape != ref.shape:
            raise ValueError(f"{name} shape {arr.shape} != {ref.shape} for {init_dir} +{hours}h")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} has non-finite values for {init_dir} +{hours}h")
    missing = set(CHANNEL_ORDER) - set(values)
    if missing:
        raise ValueError(f"missing channels {sorted(missing)} for {init_dir} +{hours}h")
    return direct, values


def _append_derived(out_path: str, template_src: str, match_param_id: int,
                    values: np.ndarray, set_param_id: int | None = None) -> None:
    """Clone a template GRIB message, set paramId + raw values, append to out_path."""
    with open(template_src, "rb") as f:
        clone = None
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            if ec.codes_get(gid, "paramId") == match_param_id:
                clone = ec.codes_clone(gid)
                ec.codes_release(gid)
                break
            ec.codes_release(gid)
    if clone is None:
        raise RuntimeError(f"no template paramId={match_param_id} in {template_src}")
    if set_param_id is not None:
        ec.codes_set(clone, "paramId", set_param_id)
    if values.size != ec.codes_get(clone, "numberOfDataPoints"):
        raise ValueError("grid size mismatch when encoding derived field")
    ec.codes_set(clone, "bitsPerValue", 24)
    ec.codes_set_values(clone, values.astype("float64").ravel())
    with open(out_path, "ab") as of:
        ec.codes_write(clone, of)
    ec.codes_release(clone)


def write_step(init_dir: str, hours: int, out_path: str) -> None:
    direct, values = compute_channels(init_dir, hours)
    tmp = out_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    direct.save(tmp)
    lead = step_file(init_dir, hours)
    _append_derived(tmp, lead, PARAM_SP, values["tcw"], set_param_id=PARAM_TCW)
    _append_derived(tmp, lead, PARAM_TP, values["tp"])
    os.replace(tmp, out_path)


def out_name(init: str, hours: int) -> str:
    return f"ifs_hres_{init}_s{hours:03d}.grib"


def parse_leads(spec: str) -> List[int]:
    """'1-90' or '1,2,3' or '6' -> list of lead hours."""
    out: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _process_lead(task):
    """Worker: write one (init, lead) GRIB. Returns a status string. Top-level so it is
    picklable by multiprocessing."""
    init_dir, hours, out_path = task
    try:
        write_step(init_dir, hours, out_path)
        return "done"
    except Exception as e:  # noqa: BLE001 - keep processing remaining leads
        logger.error("FAILED +%dh: %s", hours, e)
        return "failed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-dir", help="single IFS-HRES init cycle dir (e.g. .../IFS-HRES26/26010100)")
    p.add_argument("--init-dirs", nargs="+", help="multiple init cycle dirs (processed in one pool; "
                   "amortizes imports and maximizes parallelism). Overrides --init-dir.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--leads", default="1-90", help="lead hours, e.g. '1-90' or '1,6,12'")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes over leads")
    args = p.parse_args()

    init_dirs = args.init_dirs if args.init_dirs else ([args.init_dir] if args.init_dir else None)
    if not init_dirs:
        p.error("provide --init-dir or --init-dirs")
    os.makedirs(args.out_dir, exist_ok=True)
    leads = parse_leads(args.leads)
    logger.info("%d init dir(s): %d leads %d..%d -> %s (workers=%d)",
                len(init_dirs), len(leads), leads[0], leads[-1], args.out_dir, args.workers)

    # Build the full task list across all init dirs (one pool over everything). Skip existing
    # outputs (idempotent) and leads whose source step files are missing.
    tasks, n_skip, n_missing = [], 0, 0
    for init_dir in init_dirs:
        if not os.path.isdir(init_dir):
            logger.warning("skip missing init dir %s", init_dir)
            continue
        init = os.path.basename(init_dir.rstrip("/"))
        for hours in leads:
            out_path = os.path.join(args.out_dir, out_name(init, hours))
            if os.path.exists(out_path) and not args.overwrite:
                n_skip += 1
                continue
            if not os.path.exists(step_file(init_dir, hours)) or not os.path.exists(step_file(init_dir, hours - 1)):
                n_missing += 1  # gap; tp diff needs prev step too
                continue
            tasks.append((init_dir, hours, out_path))

    if args.workers > 1 and tasks:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(tasks))) as pool:
            results = pool.map(_process_lead, tasks, chunksize=1)
    else:
        results = [_process_lead(t) for t in tasks]

    n_done = results.count("done")
    n_fail = results.count("failed")
    logger.info("done: %d written, %d skipped, %d missing, %d failed (over %d init dirs)",
                n_done, n_skip, n_missing, n_fail, len(init_dirs))


if __name__ == "__main__":
    main()
