#!/usr/bin/env python3
"""Convert WeatherGenerator zarr output (ICON stream) to GRIB files for evalml.

Reads the ICON stream prediction from a WeatherGenerator validation zarr zip,
and writes one GRIB file per lead time per forecast initialization, in the flat
directory layout that evalml's load_forecast_data_from_grib expects.

Output layout:
    {output_dir}/{YYYYmmddHHMM}/
        {YYYYmmddHHMM}_{step:03d}.grib   (one per hourly lead time)

Each GRIB file contains all requested variables at that lead time.
TOT_PREC_1H is written as cumulative-from-start precipitation (evalml diffs it).

Usage:
    python scripts/zarr_to_grib.py \\
        --zarr /path/to/validation_chkpt00000_rank0000.zip \\
        --output /path/to/output_dir \\
        --templates /path/to/evalml/resources/inference/templates \\
        --stream ICON

Requirements: eccodes, zarr, numpy (all available in the WeatherGenerator venv)

One-time setup before evalml can read the output:
    # Download ICON-CH1 grid topology file (needed by earthkit-geo)
    mkdir -p ~/.local/share/eckit/geo/grid/icon
    curl -fL https://sites.ecmwf.int/repository/eckit/geo/grid/icon-ch/icon-ch1-c.ek \\
        -o ~/.local/share/eckit/geo/grid/icon/17643da2574959b644d254a3cd6e2bc0-b0699f374c63d05028c18c12f80a48f4.ek

    # Run evalml with COSMO eccodes definitions on PATH
    export ECCODES_DEFINITION_PATH=$(realpath evalml/.venv/share/eccodes-cosmo-resources/definitions)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import eccodes
import numpy as np
import zarr

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Variable definitions ──────────────────────────────────────────────────────

# Surface / near-surface channels.
# Maps WeatherGenerator channel name →
#   template_key: key into TEMPLATES dict below
#   shortName:    GRIB2 shortName to set
#   level:        GRIB level value
#   accumulated:  True = write as cumulative-from-start precipitation
SURFACE_CHANNELS: dict[str, dict] = {
    "T_2M": {"template_key": "heightAboveGround", "shortName": "2t", "level": 2},
    "TD_2M": {"template_key": "heightAboveGround", "shortName": "2d", "level": 2},
    "U_10M": {"template_key": "heightAboveGround", "shortName": "10u", "level": 10},
    "V_10M": {"template_key": "heightAboveGround", "shortName": "10v", "level": 10},
    "PMSL": {"template_key": "meanSea", "shortName": "prmsl", "level": 0},
    "PS": {"template_key": "surface", "shortName": "sp", "level": 0},
    "TOT_PREC_1H": {"template_key": "TOT_PREC", "shortName": "tp", "level": 0, "accumulated": True},
    "TOT_PREC": {"template_key": "TOT_PREC", "shortName": "tp", "level": 0, "accumulated": True},
}

# Pressure-level channel prefix → GRIB shortName.
# Channel names in the zarr are like T_850, QV_500, FI_250, etc.
PLEVEL_CHANNELS: dict[str, str] = {
    "T": "t",
    "QV": "q",
    "U": "u",
    "V": "v",
    "FI": "z",
}

# Template filename stems (icon-ch1 variants expected by default)
TEMPLATE_FILES = {
    "heightAboveGround": "icon-ch1-typeOfLevel=heightAboveGround.grib",
    "surface": "icon-ch1-typeOfLevel=surface.grib",
    "meanSea": "icon-ch1-typeOfLevel=meanSea.grib",
    "TOT_PREC": "icon-ch1-shortName=TOT_PREC.grib",
    "isobaricInhPa": "icon-ch1-typeOfLevel=isobaricInhPa.grib",
}


# ── Template loading ──────────────────────────────────────────────────────────


def load_templates(template_dir: Path) -> dict[str, object]:
    """Read one GRIB message from each template file; return {key: eccodes_handle}."""
    templates = {}
    for key, filename in TEMPLATE_FILES.items():
        path = template_dir / filename
        if not path.exists():
            LOG.warning("Template not found, skipping %s: %s", key, path)
            continue
        with open(path, "rb") as f:
            msg = eccodes.codes_grib_new_from_file(f)
        templates[key] = msg
        LOG.info("Loaded template: %s (%s)", key, filename)
    return templates


def release_templates(templates: dict) -> None:
    for msg in templates.values():
        eccodes.codes_release(msg)


# ── GRIB writing ──────────────────────────────────────────────────────────────


def _set_time(msg, init_dt: datetime, step_h: int, accumulated: bool = False) -> None:
    """Set forecast reference time and lead-time step on a cloned GRIB message."""
    eccodes.codes_set(msg, "dataDate", int(init_dt.strftime("%Y%m%d")))
    eccodes.codes_set(msg, "dataTime", int(init_dt.strftime("%H%M")))
    eccodes.codes_set(msg, "stepUnits", 1)  # 1 = hours
    if accumulated:
        eccodes.codes_set(msg, "startStep", 0)
    eccodes.codes_set(msg, "endStep", step_h)


def append_grib_field(
    out_path: Path,
    template_msg,
    values: np.ndarray,
    init_dt: datetime,
    step_h: int,
    short_name: str,
    level: int,
    accumulated: bool = False,
) -> None:
    """Clone template, fill in metadata + values, append to out_path."""
    msg = eccodes.codes_clone(template_msg)
    try:
        eccodes.codes_set(msg, "shortName", short_name)
        eccodes.codes_set(msg, "level", level)
        _set_time(msg, init_dt, step_h, accumulated=accumulated)
        eccodes.codes_set_values(msg, values.astype(float))
        with open(out_path, "ab") as f:
            eccodes.codes_write(msg, f)
    finally:
        eccodes.codes_release(msg)


# ── Zarr reading ──────────────────────────────────────────────────────────────


def _open_zarr(zarr_path: Path):
    """Open a WeatherGenerator zarr (zip or directory)."""
    if zarr_path.suffix == ".zip":
        store = zarr.storage.ZipStore(str(zarr_path), mode="r")
        return zarr.open(store)
    return zarr.open(str(zarr_path), mode="r")


def _iter_timesteps(stream_group):
    """Yield (time, data_slice) in chronological order, one fstep at a time.

    Processes fsteps sequentially so only one fstep's data is in memory at once.
    data_slice shape: (n_spatial, n_channels).
    """
    # Collect (time_str, fstep_key, local_mask) to sort globally
    index: list[tuple[str, str, np.datetime64]] = []
    for fstep_key in sorted(stream_group.keys(), key=int):
        fstep = stream_group[fstep_key]
        src = fstep.get("prediction") or fstep.get("target")
        raw_times = src["times"][:]
        for t in np.unique(raw_times):
            index.append((str(t), fstep_key, t))

    index.sort(key=lambda x: x[0])

    # Yield one timestep at a time, loading each fstep on demand
    current_fstep_key = None
    current_raw_times = None
    current_raw_data = None

    for _t_str, fstep_key, t in index:
        if fstep_key != current_fstep_key:
            fstep = stream_group[fstep_key]
            src = fstep.get("prediction") or fstep.get("target")
            current_raw_times = src["times"][:]
            current_raw_data = src["data"][:]
            if current_raw_data.ndim == 3:
                current_raw_data = current_raw_data[..., 0]
            current_fstep_key = fstep_key

        mask = current_raw_times == t
        yield np.datetime64(t), current_raw_data[mask]


def _init_time(stream_group) -> datetime:
    """Derive forecast initialization time from the source_interval attribute."""
    # source_interval.end is the first valid forecast time (== lead time 0)
    attrs = (
        stream_group[sorted(stream_group.keys(), key=int)[0]]
        .get("prediction", stream_group[sorted(stream_group.keys(), key=int)[0]].get("target"))
        .attrs
    )
    end_str = attrs["source_interval"]["end"]
    t = np.datetime64(end_str).astype("datetime64[ms]").astype(datetime)
    return t.replace(tzinfo=None)


# ── Per-sample conversion ─────────────────────────────────────────────────────


def convert_sample(
    stream_group,
    output_dir: Path,
    templates: dict,
    variables: list[str],
    include_pressure_levels: bool,
) -> None:
    """Convert one forecast sample (one init time) to a directory of GRIB files.

    Streams one fstep at a time so that only ~2 GB of data is in memory at once
    rather than loading all 20 fsteps (≈40 GB) upfront.
    """
    init_dt = _init_time(stream_group)
    LOG.info("─" * 60)
    LOG.info("Init time : %s", init_dt.strftime("%Y-%m-%dT%H:%M"))

    # Channel list from first available fstep
    first_key = sorted(stream_group.keys(), key=int)[0]
    src = stream_group[first_key].get("prediction") or stream_group[first_key].get("target")
    channels: list[str] = list(src.attrs["channels"])
    n_fsteps = sum(1 for _ in stream_group.keys())

    # Count total hourly timesteps (peek at times without loading data)
    n_steps_total = sum(
        len(np.unique(stream_group[k].get("prediction", stream_group[k].get("target"))["times"][:]))
        for k in stream_group.keys()
    )

    # Determine which surface vars will actually be written
    sfc_vars_out = [
        v
        for v in variables
        if v in SURFACE_CHANNELS and (v in channels or SURFACE_CHANNELS[v].get("accumulated"))
    ]

    prec_channel = next((c for c in ("TOT_PREC_1H", "TOT_PREC") if c in channels), None)
    want_precip = prec_channel is not None and prec_channel in variables
    prec_idx = channels.index(prec_channel) if want_precip else None
    cumulative_prec: np.typing.NDArray[np.float32] | None = None

    # Count pressure-level vars if requested
    plevel_vars_out: list[str] = []
    if include_pressure_levels:
        for ch in channels:
            parts = ch.rsplit("_", 1)
            if len(parts) == 2 and parts[0] in PLEVEL_CHANNELS:
                try:
                    int(parts[1])
                    plevel_vars_out.append(ch)
                except ValueError:
                    pass

    LOG.info("Fsteps    : %d  →  %d hourly steps total", n_fsteps, n_steps_total)
    LOG.info("Channels  : %d in zarr", len(channels))
    LOG.info("Sfc vars  : %s", ", ".join(sfc_vars_out) if sfc_vars_out else "(none)")
    if include_pressure_levels:
        LOG.info("PL vars   : %d (e.g. %s)", len(plevel_vars_out), ", ".join(plevel_vars_out[:4]))
    if want_precip:
        LOG.info("Precip    : %s → TOT_PREC (cumulative-from-start)", prec_channel)
    else:
        LOG.info("Precip    : not found / not requested")

    sample_dir = output_dir / init_dt.strftime("%Y%m%d%H%M")
    sample_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Output    : %s", sample_dir)

    import time as _time

    t0 = _time.monotonic()
    current_fstep_logged = None
    n_written = 0

    for step_h, (valid_t, data) in enumerate(_iter_timesteps(stream_group)):
        out_file = sample_dir / f"{init_dt.strftime('%Y%m%d%H%M')}_{step_h:03d}.grib"
        if out_file.exists():
            out_file.unlink()

        # Log once per fstep block (every 6 steps)
        fstep_block = step_h // 6 + 1
        if fstep_block != current_fstep_logged:
            elapsed = _time.monotonic() - t0
            LOG.info(
                "  fstep %2d / %d  (step %03d, valid %s)  [%.0fs elapsed]",
                fstep_block,
                n_fsteps,
                step_h,
                np.datetime_as_string(valid_t, unit="h"),
                elapsed,
            )
            current_fstep_logged = fstep_block

        # Update running precipitation accumulator
        if want_precip:
            hourly = data[:, prec_idx]
            cumulative_prec = hourly if cumulative_prec is None else cumulative_prec + hourly

        n_msgs = 0

        # ── Surface variables ─────────────────────────────────────────────
        for var in variables:
            if var not in SURFACE_CHANNELS:
                continue
            cfg = SURFACE_CHANNELS[var]
            tpl_key = cfg["template_key"]
            if tpl_key not in templates:
                LOG.warning("No template for %s (need %s), skipping", var, tpl_key)
                continue

            if cfg.get("accumulated"):
                if cumulative_prec is None:
                    continue
                values = cumulative_prec
            else:
                if var not in channels:
                    continue
                values = data[:, channels.index(var)]

            append_grib_field(
                out_path=out_file,
                template_msg=templates[tpl_key],
                values=values,
                init_dt=init_dt,
                step_h=step_h,
                short_name=cfg["shortName"],
                level=cfg["level"],
                accumulated=cfg.get("accumulated", False),
            )
            n_msgs += 1

        # ── Pressure-level variables ──────────────────────────────────────
        if include_pressure_levels and "isobaricInhPa" in templates:
            for ch in plevel_vars_out:
                parts = ch.rsplit("_", 1)
                append_grib_field(
                    out_path=out_file,
                    template_msg=templates["isobaricInhPa"],
                    values=data[:, channels.index(ch)],
                    init_dt=init_dt,
                    step_h=step_h,
                    short_name=PLEVEL_CHANNELS[parts[0]],
                    level=int(parts[1]),
                )
                n_msgs += 1

        file_kb = out_file.stat().st_size / 1024
        LOG.debug(
            "  step %03d  valid=%s  msgs=%d  size=%.0f KB",
            step_h,
            np.datetime_as_string(valid_t, unit="h"),
            n_msgs,
            file_kb,
        )
        n_written += 1

    elapsed_total = _time.monotonic() - t0
    total_mb = sum(f.stat().st_size for f in sample_dir.glob("*.grib")) / 1024**2
    LOG.info(
        "Done: %d GRIB files, %.0f MB total, %.1f s  →  %s",
        n_written,
        total_mb,
        elapsed_total,
        sample_dir,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--zarr", required=True, type=Path, help="Path to WeatherGenerator zarr zip file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Root output directory for GRIB files"
    )
    parser.add_argument(
        "--templates",
        required=True,
        type=Path,
        help="Path to evalml GRIB templates directory (resources/inference/templates/)",
    )
    parser.add_argument(
        "--stream",
        default="ICON",
        help="Stream name in the zarr to convert (default: ICON)",
    )
    parser.add_argument(
        "--samples",
        default=None,
        nargs="+",
        type=int,
        help="Sample IDs to convert. Defaults to all samples in the zarr.",
    )
    parser.add_argument(
        "--variables",
        default=["T_2M", "TD_2M", "U_10M", "V_10M", "TOT_PREC_1H", "PMSL", "PS"],
        nargs="+",
        help="Surface variable names to export (default: T_2M TD_2M U_10M V_10M TOT_PREC_1H PMSL PS)",
    )
    parser.add_argument(
        "--pressure-levels",
        action="store_true",
        default=False,
        help="Also export pressure-level variables (T, QV, U, V, FI at all levels)",
    )
    parser.add_argument(
        "--cosmo-defs",
        default=None,
        type=Path,
        help="Optional path to eccodes-cosmo-resources definitions directory",
    )
    args = parser.parse_args()

    # Optionally extend eccodes definition path with COSMO definitions
    if args.cosmo_defs and args.cosmo_defs.exists():
        import os

        existing = eccodes.codes_definition_path()
        os.environ["ECCODES_DEFINITION_PATH"] = f"{args.cosmo_defs}:{existing}"
        LOG.info("Using COSMO eccodes definitions from %s", args.cosmo_defs)

    LOG.info("Opening zarr: %s", args.zarr)
    root = _open_zarr(args.zarr)
    sample_ids = args.samples or [int(k) for k in root.keys()]
    LOG.info("Converting %d samples, stream=%s", len(sample_ids), args.stream)

    templates = load_templates(args.templates)
    if not templates:
        raise RuntimeError(f"No templates loaded from {args.templates}. Check the path.")

    args.output.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        key = str(sid)
        if key not in root:
            LOG.warning("Sample %d not found in zarr, skipping", sid)
            continue
        if args.stream not in root[key]:
            LOG.warning("Stream %s not found in sample %d, skipping", args.stream, sid)
            continue
        convert_sample(
            stream_group=root[key][args.stream],
            output_dir=args.output,
            templates=templates,
            variables=args.variables,
            include_pressure_levels=args.pressure_levels,
        )

    release_templates(templates)
    LOG.info("Done. Output written to %s", args.output)


if __name__ == "__main__":
    main()
