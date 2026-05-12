from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.paths import DEFAULT_EVAL_ROOT
from manual_inference.config import DEFAULT_EXTRA_ARGS_JSON
from manual_inference.prediction.predict import _resolve_ckpt_path as _shared_resolve_ckpt_path

DEFAULT_CKPT_ROOT = "/home/ecm5702/scratch/aifs/checkpoint"


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"Invalid empty run name from input: {name!r}")
    return cleaned


def _resolve_ckpt_path(name_ckpt: str, ckpt_root: str) -> Path:
    return Path(_shared_resolve_ckpt_path(name_ckpt, ckpt_root))


def _default_checkpoint_run_name(name_ckpt: str, ckpt_root: str) -> str:
    raw = Path(os.path.expanduser(name_ckpt))
    if raw.suffix == ".ckpt":
        exp = raw.parent.name or Path(ckpt_root).name or "checkpoint"
        return _sanitize_name(f"{exp}-{raw.stem}")
    if len(raw.parts) > 1:
        return _sanitize_name(f"{raw.parent.name}-{raw.name}")
    return _sanitize_name(f"{name_ckpt}-last")


def _prepare_run_dir(eval_root: str, run_name: str) -> Path:
    run_dir = Path(eval_root).expanduser() / _sanitize_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_metadata(run_dir: Path, payload: dict[str, Any]) -> Path:
    out = run_dir / "metadata.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out


def _copy_predictions_to_run(predictions_src: str, run_dir: Path) -> Path:
    src = Path(predictions_src).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"predictions file not found: {src}")
    dst = run_dir / "predictions.nc"
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return dst


def _run_region_plots(run_dir: Path) -> None:
    from eval._backends.region_plotting.plot_regions import run_region_plots_from_predictions

    run_region_plots_from_predictions(
        run_parent_dir=run_dir.parent,
        run_name=run_dir.name,
        predictions_filename="predictions.nc",
    )


def _run_sigma_for_checkpoint(
    *,
    ckpt_path: Path,
    out_csv: Path,
    ckpt_root: str,
    device: str,
    n_samples: int,
    validation_frequency: str,
    sigmas: str,
    run_pure_noise: bool,
    run_noised: bool,
) -> Path:
    from eval._backends.sigma_evaluator.run_sigma_evaluator import main as sigma_main

    argv = [
        "--name_exp",
        ckpt_path.parent.name,
        "--name_ckpt",
        ckpt_path.name,
        "--ckpt-root",
        ckpt_root,
        "--out_csv",
        str(out_csv),
        "--device",
        device,
        "--n_samples",
        str(n_samples),
        "--validation_frequency",
        validation_frequency,
    ]
    if sigmas:
        argv.extend(["--sigmas", sigmas])
    if run_pure_noise:
        argv.append("--run_pure_noise")
    if run_noised:
        argv.append("--run_noised")
    sigma_main(argv)
    return out_csv


def run_from_checkpoint(args: argparse.Namespace) -> Path:
    from manual_inference.prediction.predict import main as predict_main

    run_name = args.run_name or _default_checkpoint_run_name(args.name_ckpt, args.ckpt_root)
    run_dir = _prepare_run_dir(args.eval_root, run_name)
    predictions_path = run_dir / "predictions.nc"

    predict_argv = [
        "--name-ckpt",
        args.name_ckpt,
        "--ckpt-root",
        args.ckpt_root,
        "--device",
        args.device,
        "--validation-frequency",
        args.validation_frequency,
        "--extra-args-json",
        args.extra_args_json,
        "--out",
        str(predictions_path),
    ]
    if args.bundle_nc:
        predict_cmd = ["from-bundle", "--bundle-nc", args.bundle_nc]
    else:
        predict_cmd = [
            "from-dataloader",
            "--debug-from-dataloader",
            "--idx",
            str(args.idx),
            "--n-samples",
            str(args.n_samples),
            "--members",
            args.members,
        ]
    predict_main(predict_cmd + predict_argv)

    if args.run_region:
        _run_region_plots(run_dir)

    sigma_path = None
    if args.run_sigma:
        ckpt_path = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
        sigma_path = _run_sigma_for_checkpoint(
            ckpt_path=ckpt_path,
            out_csv=run_dir / "sigma_eval_table.csv",
            ckpt_root=args.ckpt_root,
            device=args.sigma_device,
            n_samples=args.sigma_n_samples,
            validation_frequency=args.validation_frequency,
            sigmas=args.sigma_values,
            run_pure_noise=args.sigma_run_pure_noise,
            run_noised=args.sigma_run_noised,
        )

    _write_metadata(
        run_dir,
        {
            "source": "checkpoint",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_dir.name,
            "name_ckpt": args.name_ckpt,
            "ckpt_root": args.ckpt_root,
            "predictions_nc": str(predictions_path),
            "sigma_csv": str(sigma_path) if sigma_path else None,
        },
    )
    return run_dir


def run_from_predictions(args: argparse.Namespace) -> Path:
    run_name = args.run_name or _sanitize_name(Path(args.predictions_nc).stem)
    run_dir = _prepare_run_dir(args.eval_root, run_name)
    predictions_path = _copy_predictions_to_run(args.predictions_nc, run_dir)

    if args.run_region:
        _run_region_plots(run_dir)

    _write_metadata(
        run_dir,
        {
            "source": "predictions",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_dir.name,
            "predictions_source_nc": str(Path(args.predictions_nc).expanduser().resolve()),
            "predictions_nc": str(predictions_path),
        },
    )
    return run_dir


def run_from_mars_expver(args: argparse.Namespace) -> Path:
    from eval._backends.region_plotting.plot_regions import (
        build_predictions_dataset_from_expver,
        save_predictions_dataset,
    )

    run_name = args.run_name or _sanitize_name(args.expver)
    run_dir = _prepare_run_dir(args.eval_root, run_name)
    predictions_path = run_dir / "predictions.nc"
    ds = build_predictions_dataset_from_expver(
        expver=args.expver,
        date=args.date,
        number=args.number,
        step=args.step,
        sfc_param=args.sfc_param,
        pl_param=args.pl_param,
        level=args.level,
        low_res_reference_grib=args.low_res_reference_grib,
        high_res_reference_grib=args.high_res_reference_grib,
    )
    save_predictions_dataset(ds, predictions_path)
    ds.close()

    if args.run_region:
        _run_region_plots(run_dir)

    _write_metadata(
        run_dir,
        {
            "source": "mars_expver",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_dir.name,
            "expver": args.expver,
            "date": args.date,
            "number": args.number,
            "step": args.step,
            "predictions_nc": str(predictions_path),
        },
    )
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified eval runner for checkpoint, predictions.nc, or MARS expver."
    )
    parser.add_argument(
        "--eval-root",
        default=DEFAULT_EVAL_ROOT,
        help=f"Root folder for eval artifacts (default: {DEFAULT_EVAL_ROOT}).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ckpt = sub.add_parser("checkpoint", help="Evaluate from a checkpoint.")
    p_ckpt.add_argument("--name-ckpt", required=True, help="Checkpoint name/path.")
    p_ckpt.add_argument("--run-name", default="", help="Optional explicit run folder name.")
    p_ckpt.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    p_ckpt.add_argument("--device", default="cuda")
    p_ckpt.add_argument("--validation-frequency", default="50h")
    p_ckpt.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_ckpt.add_argument("--bundle-nc", default="", help="If set, run prediction from bundle.")
    p_ckpt.add_argument("--idx", type=int, default=0)
    p_ckpt.add_argument("--n-samples", type=int, default=1)
    p_ckpt.add_argument("--members", default="0")
    p_ckpt.add_argument("--run-region", action="store_true", default=True)
    p_ckpt.add_argument("--skip-region", action="store_false", dest="run_region")
    p_ckpt.add_argument("--run-sigma", action="store_true", default=True)
    p_ckpt.add_argument("--skip-sigma", action="store_false", dest="run_sigma")
    p_ckpt.add_argument("--sigma-device", default="auto", choices=["auto", "cuda", "cpu"])
    p_ckpt.add_argument("--sigma-n-samples", type=int, default=10)
    p_ckpt.add_argument("--sigma-values", default="", help="Comma-separated sigma override.")
    p_ckpt.add_argument("--sigma-run-pure-noise", action="store_true")
    p_ckpt.add_argument("--sigma-run-noised", action="store_true")

    p_pred = sub.add_parser("predictions", help="Evaluate existing predictions.nc.")
    p_pred.add_argument("--predictions-nc", required=True)
    p_pred.add_argument("--run-name", default="", help="Optional explicit run folder name.")
    p_pred.add_argument("--run-region", action="store_true", default=True)
    p_pred.add_argument("--skip-region", action="store_false", dest="run_region")

    p_mars = sub.add_parser("mars-expver", help="Evaluate directly from MARS expver.")
    p_mars.add_argument("--expver", required=True)
    p_mars.add_argument("--run-name", default="", help="Optional explicit run folder name.")
    p_mars.add_argument("--date", default="20230801/20230810")
    p_mars.add_argument("--number", default="1/2/3")
    p_mars.add_argument("--step", default="48/120")
    p_mars.add_argument("--sfc-param", default="2t/10u/10v/sp")
    p_mars.add_argument("--pl-param", default="z/t/u/v")
    p_mars.add_argument("--level", default="500/850")
    p_mars.add_argument(
        "--low-res-reference-grib",
        default="eefo_reference_o96-early-august.grib",
    )
    p_mars.add_argument(
        "--high-res-reference-grib",
        default="enfo_reference_o320-early-august.grib",
    )
    p_mars.add_argument("--run-region", action="store_true", default=True)
    p_mars.add_argument("--skip-region", action="store_false", dest="run_region")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "checkpoint":
        run_dir = run_from_checkpoint(args)
    elif args.cmd == "predictions":
        run_dir = run_from_predictions(args)
    elif args.cmd == "mars-expver":
        run_dir = run_from_mars_expver(args)
    else:
        raise SystemExit(f"Unsupported command: {args.cmd}")
    print(f"Eval artifacts saved in: {run_dir}")


if __name__ == "__main__":
    main()
