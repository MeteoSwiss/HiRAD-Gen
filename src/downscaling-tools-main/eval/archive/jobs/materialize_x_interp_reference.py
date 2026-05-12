#!/usr/bin/env python3
"""Materialize an on-grid x_interp reference prediction tree.

The O96->O320 x_interp reference is the low-resolution EEFO O96 input after the
prediction exporter interpolates it to the O320 high-resolution grid. Downstream
scoreboard tools expect candidate fields under ``y_pred``, so this helper writes
new ``predictions_*.nc`` files with ``y_pred`` copied from source ``x_interp`` and
``y`` kept as the O320 truth.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import xarray as xr

PREDICTION_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
DEFAULT_DATES = (20230826, 20230827, 20230828, 20230829, 20230830)
DEFAULT_STEPS = (24, 48, 72, 96, 120)
DEFAULT_REQUIRED_WEATHER_STATES = (
    "10u",
    "10v",
    "2t",
    "sp",
    "msl",
    "t_850",
    "z_500",
)
DEFAULT_RUN_ID = "reference_x_interp_o96_o320"


def _parse_csv_ints(raw: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text or text.upper() == "DEFAULT":
        return default
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_strings(raw: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text or text.upper() == "DEFAULT":
        return default
    return tuple(part.strip() for part in text.split(",") if part.strip())


def prediction_file_scope(path: Path) -> tuple[int, int]:
    match = PREDICTION_RE.match(path.name)
    if not match:
        raise ValueError(f"not a canonical prediction filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def discover_prediction_files(predictions_dir: Path) -> list[Path]:
    files = [path for path in sorted(predictions_dir.glob("predictions_*.nc")) if PREDICTION_RE.match(path.name)]
    if not files:
        raise FileNotFoundError(f"no predictions_*.nc files found in {predictions_dir}")
    return files


def validate_source_predictions(
    predictions_dir: Path,
    *,
    expected_dates: Iterable[int] = DEFAULT_DATES,
    expected_steps: Iterable[int] = DEFAULT_STEPS,
    required_weather_states: Iterable[str] = DEFAULT_REQUIRED_WEATHER_STATES,
) -> dict[str, object]:
    files = discover_prediction_files(predictions_dir)
    expected_date_set = {int(value) for value in expected_dates}
    expected_step_set = {int(value) for value in expected_steps}
    required_state_set = {str(value) for value in required_weather_states}

    dates: set[int] = set()
    steps: set[int] = set()
    for path in files:
        date, step = prediction_file_scope(path)
        dates.add(date)
        steps.add(step)

    expected_count = len(expected_date_set) * len(expected_step_set)
    if len(files) != expected_count:
        raise ValueError(f"expected {expected_count} prediction files, found {len(files)} in {predictions_dir}")
    if dates != expected_date_set:
        raise ValueError(f"expected dates {sorted(expected_date_set)}, found {sorted(dates)}")
    if steps != expected_step_set:
        raise ValueError(f"expected steps {sorted(expected_step_set)}, found {sorted(steps)}")

    with xr.open_dataset(files[0], cache=False) as ds:
        for required_var in ("x_interp", "y", "weather_state", "lon_hres", "lat_hres"):
            if required_var not in ds:
                raise ValueError(f"{files[0]} is missing required variable/coord {required_var!r}")
        weather_states = {str(value) for value in ds["weather_state"].values.tolist()}
        missing_states = sorted(required_state_set - weather_states)
        if missing_states:
            raise ValueError(f"{files[0]} is missing required weather states: {missing_states}")
        member_count = int(ds.sizes.get("ensemble_member") or ds.sizes.get("member") or 1)
        x_interp_dims = tuple(ds["x_interp"].dims)
        y_dims = tuple(ds["y"].dims)
        x_interp_shape = tuple(int(value) for value in ds["x_interp"].shape)
        y_shape = tuple(int(value) for value in ds["y"].shape)
        if x_interp_shape != y_shape:
            raise ValueError(f"x_interp shape {x_interp_shape} does not match y shape {y_shape} in {files[0]}")

    return {
        "predictions_dir": str(predictions_dir),
        "prediction_files": len(files),
        "dates": sorted(dates),
        "steps_hours": sorted(steps),
        "required_weather_states": sorted(required_state_set),
        "weather_states": sorted(weather_states),
        "member_count": member_count,
        "x_interp_dims": list(x_interp_dims),
        "y_dims": list(y_dims),
        "x_interp_shape": list(x_interp_shape),
    }


def _copy_variable_if_present(source: xr.Dataset, target: xr.Dataset, name: str) -> xr.Dataset:
    if name in source and name not in target:
        target[name] = source[name]
    return target


def materialize_file(source_path: Path, out_path: Path, *, source_predictions_dir: Path, run_id: str) -> None:
    with xr.open_dataset(source_path, cache=False) as ds:
        for required_var in ("x_interp", "y"):
            if required_var not in ds:
                raise ValueError(f"{source_path} is missing required variable {required_var!r}")

        out = xr.Dataset(coords={name: coord for name, coord in ds.coords.items()})
        out["y_pred"] = ds["x_interp"]
        out["y"] = ds["y"]
        out["x_interp"] = ds["x_interp"]
        for name in (
            "date",
            "init_date",
            "lead_step_hours",
            "lon_hres",
            "lat_hres",
            "area_weight",
            "source_bundle",
        ):
            out = _copy_variable_if_present(ds, out, name)

        attrs = dict(ds.attrs)
        attrs.update(
            {
                "run_id": run_id,
                "reference_type": "x_interp_o96_to_o320",
                "source_predictions_dir": str(source_predictions_dir),
                "source_prediction_file": str(source_path),
                "scoreboard_candidate_field": "y_pred",
                "scoreboard_candidate_field_source": "x_interp",
                "checkpoint_id": "",
                "checkpoint_path": "",
                "sampling_config_json": "{}",
            }
        )
        out.attrs = attrs

        out_path.parent.mkdir(parents=True, exist_ok=True)
        data_vars = set(out.data_vars)
        encoding = {
            name: {"zlib": True, "complevel": 1, "shuffle": True}
            for name in data_vars
            if out[name].dtype.kind in {"f", "i", "u"}
        }
        try:
            out.to_netcdf(out_path, encoding=encoding)
        except ValueError:
            out.to_netcdf(out_path)
        finally:
            out.close()


def materialize_reference(
    source_predictions_dir: Path,
    out_root: Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    expected_dates: Iterable[int] = DEFAULT_DATES,
    expected_steps: Iterable[int] = DEFAULT_STEPS,
    required_weather_states: Iterable[str] = DEFAULT_REQUIRED_WEATHER_STATES,
    overwrite: bool = False,
) -> dict[str, object]:
    source_predictions_dir = source_predictions_dir.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    out_predictions_dir = out_root / "predictions"

    validation = validate_source_predictions(
        source_predictions_dir,
        expected_dates=expected_dates,
        expected_steps=expected_steps,
        required_weather_states=required_weather_states,
    )
    source_files = discover_prediction_files(source_predictions_dir)

    if out_predictions_dir.exists() and any(out_predictions_dir.glob("predictions_*.nc")) and not overwrite:
        raise FileExistsError(f"output predictions already exist: {out_predictions_dir}; pass --overwrite to replace")
    out_predictions_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for source_path in source_files:
        out_path = out_predictions_dir / source_path.name
        if out_path.exists():
            if overwrite:
                out_path.unlink()
            else:
                raise FileExistsError(f"output file exists: {out_path}")
        materialize_file(source_path, out_path, source_predictions_dir=source_predictions_dir, run_id=run_id)
        written.append(str(out_path))

    summary = {
        "run_id": run_id,
        "reference_type": "x_interp_o96_to_o320",
        "source_predictions_dir": str(source_predictions_dir),
        "out_root": str(out_root),
        "out_predictions_dir": str(out_predictions_dir),
        "files_written": len(written),
        "written_files": written,
        "validation": validation,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "X_INTERP_REFERENCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize x_interp as y_pred for O96->O320 reference evaluation.")
    parser.add_argument("--source-predictions-dir", required=True)
    parser.add_argument("--out-root", default="/home/ecm5702/perm/eval/reference_x_interp_o96_o320")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--dates", default="DEFAULT")
    parser.add_argument("--steps", default="DEFAULT")
    parser.add_argument("--required-weather-states", default="DEFAULT")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_predictions_dir = Path(args.source_predictions_dir)
    out_root = Path(args.out_root)
    dates = _parse_csv_ints(args.dates, default=DEFAULT_DATES)
    steps = _parse_csv_ints(args.steps, default=DEFAULT_STEPS)
    required_weather_states = _parse_csv_strings(
        args.required_weather_states,
        default=DEFAULT_REQUIRED_WEATHER_STATES,
    )
    if args.validate_only:
        summary = validate_source_predictions(
            source_predictions_dir.expanduser().resolve(),
            expected_dates=dates,
            expected_steps=steps,
            required_weather_states=required_weather_states,
        )
    else:
        summary = materialize_reference(
            source_predictions_dir,
            out_root,
            run_id=args.run_id,
            expected_dates=dates,
            expected_steps=steps,
            required_weather_states=required_weather_states,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
