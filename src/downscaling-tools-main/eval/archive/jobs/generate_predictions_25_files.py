#!/usr/bin/env python3
"""Generate predictions from input bundles — DEPRECATED.

This script is superseded by ``python -m eval.predict.main`` which provides
the same CLI interface with proper date handling, modular architecture, and
schema validation.  This file is kept for reference and backward compatibility.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.dataset import OUTPUT_WEATHER_STATE_MODE_CHOICES
from manual_inference.prediction.predict import (
    DEFAULT_EXTRA_ARGS_JSON,
    _compute_x_interp_for_export,
    _get_parallel_info,
    _init_model_comm_group,
    _load_objects,
    _predict_from_bundle,
    _resolve_ckpt_path,
    _resolve_device,
)

BUNDLE_RE = re.compile(
    r"date(?P<date>\d{8})_time\d{4}_mem(?P<member>\d+)_step(?P<step>\d{3})h_input_bundle\.nc$"
)
RANK0_WRITE_DONE_SUFFIX = ".rank0-write-done"
RANK0_WRITE_FAILED_SUFFIX = ".rank0-write-failed"
DEFAULT_RANK0_WRITE_WAIT_SECONDS = 7200


@dataclass(frozen=True)
class BundleKey:
    date: str
    step: int
    member: int


def parse_bundle_key(path: Path) -> BundleKey | None:
    m = BUNDLE_RE.search(path.name)
    if not m:
        return None
    return BundleKey(date=m.group("date"), step=int(m.group("step")), member=int(m.group("member")))


def discover_bundles(input_root: Path, *, recursive: bool = True) -> dict[BundleKey, Path]:
    out: dict[BundleKey, Path] = {}
    paths = input_root.rglob("*_input_bundle.nc") if recursive else input_root.glob("*_input_bundle.nc")
    for p in sorted(paths):
        key = parse_bundle_key(p)
        if key is None:
            continue
        if key not in out or p.stat().st_mtime >= out[key].stat().st_mtime:
            out[key] = p
    return out


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def parse_output_weather_states(raw: str) -> list[str] | None:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    return requested or None


def _member_field_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported {name} member field shape {arr.shape}; expected 2D after singleton sample/member axes.")
    return arr


def _call_accepts_kwarg(fn, name: str) -> bool:
    params = inspect.signature(fn).parameters
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _distributed_barrier(*, args_device: str, local_rank: int, world_size: int) -> None:
    if world_size <= 1 or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    sync_device_ids = [local_rank] if args_device == "cuda" and torch.cuda.is_available() else None
    torch.distributed.barrier(device_ids=sync_device_ids)


def _rank0_done_marker(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + RANK0_WRITE_DONE_SUFFIX)


def _rank0_failed_marker(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + RANK0_WRITE_FAILED_SUFFIX)


def _clear_rank0_write_markers(out_path: Path) -> None:
    for marker in (_rank0_done_marker(out_path), _rank0_failed_marker(out_path)):
        if marker.exists():
            marker.unlink()


def _write_rank0_failure_marker(out_path: Path, exc: BaseException) -> None:
    _rank0_failed_marker(out_path).write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )


def _mark_rank0_write_complete(out_path: Path) -> None:
    _rank0_done_marker(out_path).write_text("ok\n", encoding="utf-8")


def _wait_for_rank0_write(
    *,
    out_path: Path,
    global_rank: int,
    timeout_seconds: int,
    poll_seconds: float = 5.0,
) -> None:
    if global_rank == 0:
        return

    done_marker = _rank0_done_marker(out_path)
    failed_marker = _rank0_failed_marker(out_path)
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if done_marker.exists():
            return
        if failed_marker.exists():
            detail = failed_marker.read_text(encoding="utf-8").strip()
            raise SystemExit(
                f"Rank-0 write failed for {out_path}: {detail or 'unknown error'}"
            )
        time.sleep(poll_seconds)

    raise TimeoutError(
        f"Timed out waiting for rank 0 to finish writing {out_path} "
        f"after {timeout_seconds}s."
    )


def _destroy_process_group() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 25 predictions_YYYYMMDD_stepXXX.nc files from bundle inputs.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--input-root-mode",
        choices=["auto", "lane_canonical_root", "rebuilt_truth_bundle_root"],
        default="auto",
        help=(
            "Interpretation of --input-root. lane_canonical_root disables recursive bundle discovery "
            "so nested bundle caches under raw source trees are ignored."
        ),
    )
    ap.add_argument("--ckpt-id", default="")
    ap.add_argument(
        "--name-ckpt",
        default="",
        help="Explicit checkpoint path under --ckpt-root or an absolute .ckpt path. Overrides --ckpt-id.",
    )
    ap.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-gpus-per-model", type=int, default=1)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--validation-frequency", default="50h")
    ap.add_argument("--members", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--steps", default="24,48,72,96,120")
    ap.add_argument("--dates", default="20230826,20230827,20230828,20230829,20230830")
    ap.add_argument(
        "--bundle-pairs",
        default="",
        help=(
            "Explicit date:step pairs instead of the cartesian product of --dates × --steps. "
            "Format: '20230828:24,20230829:48,...'. When set, --dates and --steps are ignored "
            "for file selection (but still parsed for backward compat)."
        ),
    )
    ap.add_argument(
        "--extra-args-json",
        default=DEFAULT_EXTRA_ARGS_JSON,
    )
    ap.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="surface-plus-core-pl",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    ap.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    ap.add_argument(
        "--slim-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write only canonical x/x_interp/y/y_pred ensemble arrays and skip duplicate "
            "x_*/y_*/y_pred_* member views. Defaults to slim output."
        ),
    )
    ap.add_argument("--manifest", default="")
    ap.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Deprecated in new stack: y truth is required for predictions output.",
    )
    ap.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow date/step outputs with no target_hres_* truth by writing all-NaN y. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    ap.add_argument(
        "--allow-existing-out-dir",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    ap.add_argument(
        "--allow-overwrite-existing-files",
        action="store_true",
        help="Allow overwriting existing predictions_*.nc files.",
    )
    ap.add_argument(
        "--rank0-write-wait-seconds",
        type=int,
        default=DEFAULT_RANK0_WRITE_WAIT_SECONDS,
        help=(
            "How long nonzero ranks wait for rank 0 to finish the per-file NetCDF write "
            "before aborting."
        ),
    )
    args = ap.parse_args()

    if args.allow_missing_target:
        raise SystemExit(
            "--allow-missing-target is no longer supported in the new stack. "
            "Bundles must include target_hres_* so y is always present."
        )

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)

    global_rank, local_rank, world_size = _get_parallel_info()
    dir_check_error = None
    if global_rank == 0:
        try:
            if out_dir.exists():
                if not args.allow_existing_out_dir and any(out_dir.iterdir()):
                    raise SystemExit(
                        f"Output directory already exists and is not empty: {out_dir}. "
                        "Use a fresh run folder or pass --allow-existing-out-dir explicitly."
                    )
            else:
                out_dir.mkdir(parents=True, exist_ok=False)
        except BaseException as exc:
            dir_check_error = exc

    _distributed_barrier(args_device=args.device, local_rank=local_rank, world_size=world_size)

    if dir_check_error is not None:
        raise dir_check_error

    try:
        members = parse_int_list(args.members)
        steps = parse_int_list(args.steps)
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]

        # --bundle-pairs overrides the cartesian product of dates × steps
        if args.bundle_pairs:
            date_step_pairs = []
            for token in args.bundle_pairs.split(","):
                token = token.strip()
                if not token:
                    continue
                d, s = token.split(":")
                date_step_pairs.append((d.strip(), int(s.strip())))
            dates_used = sorted({d for d, _ in date_step_pairs})
            steps_used = sorted({s for _, s in date_step_pairs})
            print(f"Using --bundle-pairs: {len(date_step_pairs)} date/step pairs "
                  f"(dates={dates_used}, steps={steps_used})")
        else:
            date_step_pairs = [(d, s) for d in dates for s in steps]

        recursive_bundle_search = args.input_root_mode != "lane_canonical_root"
        bundle_map = discover_bundles(input_root, recursive=recursive_bundle_search)
        if not bundle_map:
            raise SystemExit(f"No bundle files found in {input_root}")

        required = [BundleKey(d, s, m) for d, s in date_step_pairs for m in members]
        missing = [k for k in required if k not in bundle_map]
        if missing:
            head = ", ".join(f"{m.date}/step{m.step}/mem{m.member}" for m in missing[:15])
            raise SystemExit(f"Missing {len(missing)} required bundle(s). First missing: {head}")

        if args.name_ckpt:
            ckpt_path = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
        elif args.ckpt_id:
            ckpt_path = os.path.join(args.ckpt_root, args.ckpt_id, "last.ckpt")
        else:
            raise SystemExit("Pass either --name-ckpt or --ckpt-id.")
        extra_args = json.loads(args.extra_args_json) if args.extra_args_json else {}
        output_weather_states = parse_output_weather_states(args.output_weather_states)

        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "Requested --device cuda, but CUDA is not available on this host. "
                "Refusing to fall back to CPU for diffusion sampling."
            )

        device = _resolve_device(args.device, local_rank)
        if str(device).startswith("cuda"):
            torch.cuda.set_device(int(str(device).split(":")[1]))
        if args.num_gpus_per_model > 1 and world_size != args.num_gpus_per_model:
            raise SystemExit(
                f"Expected world_size={args.num_gpus_per_model} for model-parallel inference, got {world_size}. "
                "Launch with matching srun --ntasks."
            )
        model_comm_group = _init_model_comm_group(device, global_rank, world_size)

        inference_model, datamodule, _, _ = _load_objects(
            ckpt_path=ckpt_path,
            device=device,
            validation_frequency=args.validation_frequency,
            precision=args.precision,
            num_gpus_per_model_override=args.num_gpus_per_model,
        )

        manifest_lines = ["date,step,member,bundle_path,predictions_path"]
        total_files = len(date_step_pairs)
        done_files = 0
        keep_outputs = global_rank == 0

        for date, step in date_step_pairs:
                out_path = out_dir / f"predictions_{date}_step{step:03d}.nc"
                x_members: list[np.ndarray] = []
                y_members: list[np.ndarray | None] = []
                yp_members: list[np.ndarray] = []
                source_paths: list[str] = []
                members_missing_target: list[int] = []

                lon_lres = lat_lres = lon_hres = lat_hres = weather_states = None
                for m in members:
                    key = BundleKey(date, step, m)
                    bundle_path = bundle_map[key]
                    x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, _dates = _predict_from_bundle(
                        inference_model=inference_model,
                        datamodule=datamodule,
                        device=device,
                        bundle_nc=str(bundle_path),
                        member_index=0,
                        extra_args=extra_args,
                        precision=args.precision,
                        model_comm_group=model_comm_group,
                        output_weather_state_mode=args.output_weather_state_mode,
                        output_weather_states=output_weather_states,
                    )

                    x_members.append(_member_field_array("x", x))
                    if keep_outputs:
                        y_members.append(None if y is None else _member_field_array("y", y))
                        if y is None:
                            members_missing_target.append(m)
                        yp_members.append(_member_field_array("y_pred", y_pred))
                        source_paths.append(str(bundle_path))

                    del x, y, y_pred

                x_stack = np.stack(x_members, axis=0)[None, ...]
                x_interp_stack = None
                yp_stack = None
                y_stack = None
                used_missing_target_unsafe = False
                ds = None
                try:
                    try:
                        x_interp_stack = _compute_x_interp_for_export(
                            inference_model=inference_model,
                            x=x_stack,
                            device=device,
                            model_comm_group=model_comm_group,
                        )
                    except RuntimeError as exc:
                        if "cannot export x_interp" not in str(exc):
                            raise
                        x_interp_stack = None
                        print(
                            f"WARNING date={date} step={step}: skipping x_interp export because the "
                            f"inference model does not expose interpolation hooks ({exc})."
                        )
                    if keep_outputs:
                        _clear_rank0_write_markers(out_path)
                        yp_stack = np.stack(yp_members, axis=0)[None, ...]
                        if any(y is not None for y in y_members):
                            template = next(y for y in y_members if y is not None)
                            y_filled = [
                                (y if y is not None else np.full_like(template, np.nan, dtype=np.float32))
                                for y in y_members
                            ]
                            y_stack = np.stack(y_filled, axis=0)[None, ...]
                            if members_missing_target:
                                print(
                                    f"WARNING date={date} step={step}: missing target y for members {members_missing_target}; "
                                    "filled with NaN to keep y present in predictions file."
                                )
                        else:
                            if not args.allow_missing_target_unsafe:
                                raise SystemExit(
                                    f"No target truth (y) extracted for date={date} step={step}. "
                                    "Rebuild bundles with target_hres_* fields."
                                )
                            y_stack = np.full_like(yp_stack, np.nan, dtype=np.float32)
                            used_missing_target_unsafe = True
                            print(
                                f"WARNING date={date} step={step}: no target y for any selected member; "
                                "filled all y with NaN because --allow-missing-target-unsafe was set. "
                                "Treat this file as prediction-only and non-canonical for truth-aware evaluation."
                            )

                        build_dataset_kwargs = {
                            "x": x_stack,
                            "y": y_stack,
                            "y_pred": yp_stack,
                            "lon_lres": lon_lres,
                            "lat_lres": lat_lres,
                            "lon_hres": lon_hres,
                            "lat_hres": lat_hres,
                            "weather_states": weather_states,
                            "dates": None,
                            "member_ids": members,
                            "include_member_views": not args.slim_output,
                        }
                        if _call_accepts_kwarg(build_predictions_dataset, "x_interp"):
                            build_dataset_kwargs["x_interp"] = x_interp_stack
                        ds = build_predictions_dataset(**build_dataset_kwargs)
                        if used_missing_target_unsafe:
                            ds.attrs["missing_target_policy"] = "all_nan_due_to_allow_missing_target_unsafe"
                        ds = ds.assign_coords(sample=[0])
                        ds["init_date"] = xr.DataArray([date], dims=("sample",))
                        ds["lead_step_hours"] = xr.DataArray([step], dims=("sample",))
                        ds["source_bundle"] = xr.DataArray(
                            np.array([source_paths], dtype=object),
                            dims=("sample", "ensemble_member"),
                        )

                        ds.attrs["checkpoint_id"] = args.ckpt_id
                        ds.attrs["checkpoint_path"] = ckpt_path
                        ds.attrs["sampling_config_json"] = args.extra_args_json
                        ds.attrs["validation_frequency"] = args.validation_frequency
                        ds.attrs["target_members_with_data"] = int(sum(y is not None for y in y_members))
                        ds.attrs["target_members_missing_data"] = int(len(members_missing_target))
                        ds.attrs["target_missing_member_ids"] = ",".join(str(m) for m in members_missing_target)
                        ds.attrs["target_weather_state_count"] = int(ds.sizes["weather_state"])
                        ds.attrs["output_weather_state_mode"] = args.output_weather_state_mode
                        ds.attrs["output_weather_states"] = ",".join(
                            str(v) for v in ds["weather_state"].values.tolist()
                        )
                        ds.attrs["x_interp_exported"] = int(x_interp_stack is not None)
                        ds.attrs["slim_output"] = int(bool(args.slim_output))

                        if out_path.exists():
                            if args.allow_overwrite_existing_files:
                                out_path.unlink()
                            else:
                                raise SystemExit(
                                    f"Refusing to overwrite existing prediction file: {out_path}. "
                                    "Use a fresh output directory or pass --allow-overwrite-existing-files explicitly."
                                )

                        ds.to_netcdf(out_path)
                        _mark_rank0_write_complete(out_path)

                        for m, src in zip(members, source_paths):
                            manifest_lines.append(f"{date},{step},{m},{src},{out_path}")

                        done_files += 1
                        print(f"[{done_files}/{total_files}] wrote {out_path}")
                except BaseException as exc:
                    if keep_outputs:
                        _write_rank0_failure_marker(out_path, exc)
                    raise
                finally:
                    if ds is not None:
                        ds.close()
                    del (
                        ds,
                        x_stack,
                        x_interp_stack,
                        y_stack,
                        yp_stack,
                        x_members,
                        y_members,
                        yp_members,
                        source_paths,
                        members_missing_target,
                    )

                gc.collect()
                _wait_for_rank0_write(
                    out_path=out_path,
                    global_rank=global_rank,
                    timeout_seconds=args.rank0_write_wait_seconds,
                )

        manifest_path = Path(args.manifest) if args.manifest else (out_dir / "predictions_manifest.csv")
        if global_rank == 0:
            manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
            print(f"Wrote manifest: {manifest_path}")
    finally:
        _destroy_process_group()


if __name__ == "__main__":
    main()
