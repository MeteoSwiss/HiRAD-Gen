"""CLI entry point for modular bundle-based prediction generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from manual_inference.prediction.dataset import OUTPUT_WEATHER_STATE_MODE_CHOICES
from manual_inference.prediction.predict import DEFAULT_EXTRA_ARGS_JSON, _get_parallel_info, _resolve_ckpt_path

from .bundle_manager import discover_bundles, resolve_date_step_pairs
from .distributed_io import Rank0FileWriter, _destroy_process_group, _distributed_barrier
from .inference_engine import predict_ensemble_members
from .model_loader import load_inference_model
from .output_writer import prediction_output_path, write_predictions_file
from .types import BundleKey, PredictionConfig


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers."""

    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def parse_output_weather_states(raw: str) -> list[str] | None:
    """Parse an explicit weather-state override."""

    requested = [token.strip() for token in raw.split(",") if token.strip()]
    return requested or None


def create_parser() -> argparse.ArgumentParser:
    """Create the backward-compatible CLI parser."""

    parser = argparse.ArgumentParser(
        description="Generate 25 predictions_YYYYMMDD_stepXXX.nc files from bundle inputs."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--input-root-mode",
        choices=["auto", "lane_canonical_root", "rebuilt_truth_bundle_root"],
        default="auto",
        help=(
            "Interpretation of --input-root. lane_canonical_root disables recursive bundle discovery "
            "so nested bundle caches under raw source trees are ignored."
        ),
    )
    parser.add_argument("--ckpt-id", default="")
    parser.add_argument(
        "--name-ckpt",
        default="",
        help="Explicit checkpoint path under --ckpt-root or an absolute .ckpt path. Overrides --ckpt-id.",
    )
    parser.add_argument("--ckpt-root", default="/home/ecm5702/scratch/aifs/checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--validation-frequency", default="50h")
    parser.add_argument("--members", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--steps", default="24,48,72,96,120")
    parser.add_argument("--dates", default="20230826,20230827,20230828,20230829,20230830")
    parser.add_argument(
        "--bundle-pairs",
        default="",
        help=(
            "Explicit date:step pairs instead of the cartesian product of --dates × --steps. "
            "Format: '20230828:24,20230829:48,...'. When set, --dates and --steps are ignored "
            "for file selection (but still parsed for backward compat)."
        ),
    )
    parser.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    parser.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="surface-plus-core-pl",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    parser.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    parser.add_argument(
        "--slim-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write only canonical x/x_interp/y/y_pred ensemble arrays and skip duplicate "
            "x_*/y_*/y_pred_* member views. Defaults to slim output."
        ),
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Deprecated in new stack: y truth is required for predictions output.",
    )
    parser.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow date/step outputs with no target_hres_* truth by writing all-NaN y. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    parser.add_argument(
        "--allow-existing-out-dir",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--allow-overwrite-existing-files",
        action="store_true",
        help="Allow overwriting existing predictions_*.nc files.",
    )
    parser.add_argument(
        "--rank0-write-wait-seconds",
        type=int,
        default=7200,
        help=(
            "How long nonzero ranks wait for rank 0 to finish the per-file NetCDF write "
            "before aborting."
        ),
    )
    return parser


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.name_ckpt:
        return Path(_resolve_ckpt_path(args.name_ckpt, args.ckpt_root))
    if args.ckpt_id:
        return Path(args.ckpt_root) / args.ckpt_id / "last.ckpt"
    raise SystemExit("Pass either --name-ckpt or --ckpt-id.")


def _prepare_output_dir(out_dir: Path, *, allow_existing: bool, global_rank: int) -> None:
    if global_rank != 0:
        return
    if out_dir.exists():
        if not allow_existing and any(out_dir.iterdir()):
            raise SystemExit(
                f"Output directory already exists and is not empty: {out_dir}. "
                "Use a fresh run folder or pass --allow-existing-out-dir explicitly."
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)


def _write_manifest(manifest_path: Path, rows: list[tuple[str, int, int, str, str]]) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "step", "member", "bundle_path", "predictions_path"])
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the modular predictions workflow."""

    args = create_parser().parse_args(argv)
    if args.allow_missing_target:
        raise SystemExit(
            "--allow-missing-target is no longer supported in the new stack. "
            "Bundles must include target_hres_* so y is always present."
        )

    input_dir = Path(args.input_root)
    out_dir = Path(args.out_dir)
    initial_global_rank, initial_local_rank, initial_world_size = _get_parallel_info()
    _prepare_output_dir(
        out_dir,
        allow_existing=args.allow_existing_out_dir,
        global_rank=initial_global_rank,
    )
    _distributed_barrier(
        args_device=args.device,
        local_rank=initial_local_rank,
        world_size=initial_world_size,
    )

    config = PredictionConfig(
        checkpoint_path=_resolve_checkpoint_path(args),
        input_dir=input_dir,
        output_dir=out_dir,
        device=args.device,
        precision=args.precision,
        num_gpus_per_model=args.num_gpus_per_model,
        validation_frequency=args.validation_frequency,
        extra_args_json=args.extra_args_json,
        output_weather_state_mode=args.output_weather_state_mode,
        output_weather_states=parse_output_weather_states(args.output_weather_states),
        slim_output=bool(args.slim_output),
        allow_overwrite=bool(args.allow_overwrite_existing_files),
        rank0_write_wait_seconds=int(args.rank0_write_wait_seconds),
        checkpoint_id=args.ckpt_id,
        input_root_mode=args.input_root_mode,
        allow_existing_out_dir=bool(args.allow_existing_out_dir),
        allow_missing_target_unsafe=bool(args.allow_missing_target_unsafe),
        manifest_path=Path(args.manifest) if args.manifest else None,
        members=parse_int_list(args.members),
        steps=parse_int_list(args.steps),
        dates=[token.strip() for token in args.dates.split(",") if token.strip()],
        bundle_pairs=args.bundle_pairs,
    )

    recursive = config.input_root_mode != "lane_canonical_root"
    bundle_map = discover_bundles(config.input_dir, recursive=recursive)
    if not bundle_map:
        raise SystemExit(f"No bundle files found in {config.input_dir}")
    date_step_pairs = resolve_date_step_pairs(
        bundle_map,
        config.dates,
        config.steps,
        config.bundle_pairs,
    )
    if config.bundle_pairs and initial_global_rank == 0:
        dates_used = sorted({date for date, _ in date_step_pairs})
        steps_used = sorted({step for _, step in date_step_pairs})
        print(
            f"Using --bundle-pairs: {len(date_step_pairs)} date/step pairs "
            f"(dates={dates_used}, steps={steps_used})"
        )

    required = [BundleKey(date, step, member) for date, step in date_step_pairs for member in config.members]
    missing = [key for key in required if key not in bundle_map]
    if missing:
        preview = ", ".join(f"{item.date}/step{item.step}/mem{item.member}" for item in missing[:15])
        raise SystemExit(f"Missing {len(missing)} required bundle(s). First missing: {preview}")

    manifest_rows: list[tuple[str, int, int, str, str]] = []
    total_files = len(date_step_pairs)
    done_files = 0
    try:
        (
            inference_model,
            datamodule,
            extra_args,
            device,
            model_comm_group,
            global_rank,
            _,
            _,
        ) = load_inference_model(config)
        writer = Rank0FileWriter(
            global_rank=global_rank,
            timeout_seconds=config.rank0_write_wait_seconds,
        )
        for date, step in date_step_pairs:
            ensemble = predict_ensemble_members(
                inference_model,
                datamodule,
                device,
                bundle_map,
                date,
                step,
                config.members,
                extra_args=extra_args,
                precision=config.precision,
                model_comm_group=model_comm_group,
                allow_missing_target_unsafe=config.allow_missing_target_unsafe,
                output_weather_state_mode=config.output_weather_state_mode,
                output_weather_states=config.output_weather_states,
                keep_outputs=(global_rank == 0),
            )
            out_path = prediction_output_path(config.output_dir, date, step)
            write_predictions_file(ensemble, out_path, config, writer)
            if global_rank == 0:
                done_files += 1
                for member, bundle_path in zip(config.members, ensemble.source_bundle_paths):
                    manifest_rows.append((date, step, member, bundle_path, str(out_path)))
                print(f"[{done_files}/{total_files}] wrote {out_path}")

        manifest_path = config.manifest_path or (config.output_dir / "predictions_manifest.csv")
        if global_rank == 0:
            _write_manifest(manifest_path, manifest_rows)
            print(f"Wrote manifest: {manifest_path}")
    finally:
        _destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
