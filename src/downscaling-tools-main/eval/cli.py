"""Unified CLI entry point for the evaluation framework.

Canonical invocation: python -m eval.cli <subcommand>

Subcommands:
    run         Full pipeline: predict + evaluate + scoreboard
    predict     Generate predictions only (subprocess call to eval.predict.main)
    evaluate    Run evaluators on existing predictions
    scoreboard  Generate scoreboard from existing evaluation results
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.config.loader import load_host, load_lane
from eval.paths import resolve_eval_root

LOG = logging.getLogger(__name__)

ALL_EVALUATORS = [
    "tc", "spectra", "surface", "region_plot",
    "sigma", "mechanistic", "intermediate",
    "spectra_ecmwf", "mlflow",
]

DEFAULT_HOST = "atos_ac"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    parser.add_argument(
        "--lane", required=True,
        help="Lane name (must match a YAML file in eval/config/lanes/).",
    )
    parser.add_argument(
        "--host", default=None,
        help=f"Host config name (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print resolved config as JSON and exit without running.",
    )


def _add_evaluator_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add --only and --include-diagnostics for evaluator selection."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", default=None,
        help="Comma-separated list of evaluators to run (overrides lane YAML groups).",
    )
    group.add_argument(
        "--include-diagnostics", action="store_true", default=False,
        help="Run default + diagnostics evaluator groups from lane YAML.",
    )


def _add_lane_override_args(parser: argparse.ArgumentParser) -> None:
    """Add lane-overridable args.  All use default=None for precedence detection."""
    parser.add_argument(
        "--members", default=None,
        help="Comma-separated member indices (e.g. 1,2,3). Overrides lane YAML predict.members.",
    )
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated forecast steps (e.g. 24,48). Overrides lane YAML predict.steps.",
    )
    parser.add_argument(
        "--dates", default=None,
        help="Comma-separated dates YYYYMMDD (e.g. 20230826,20230827). Overrides lane YAML predict.dates.",
    )


def _add_prepare_args(parser: argparse.ArgumentParser) -> None:
    """Add truth-aware bundle-building args."""
    parser.add_argument(
        "--source-grib-root", default=None,
        help="Root directory of source GRIB files for truth-aware bundle building.",
    )
    parser.add_argument(
        "--bundle-dir", default=None,
        help="Output directory for built bundles (default: <output-dir>/bundles).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="Unified CLI for the evaluation framework.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Full pipeline: predict + evaluate + scoreboard.")
    _add_common_args(p_run)
    p_run.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    _add_evaluator_filter_args(p_run)
    _add_lane_override_args(p_run)
    _add_prepare_args(p_run)
    p_run.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )

    # --- predict ---
    p_predict = subparsers.add_parser("predict", help="Generate predictions only.")
    _add_common_args(p_predict)
    p_predict.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    _add_lane_override_args(p_predict)
    _add_prepare_args(p_predict)

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="Build truth-aware bundles only (no prediction).")
    _add_common_args(p_prepare)
    _add_lane_override_args(p_prepare)
    p_prepare.add_argument(
        "--source-grib-root", required=True,
        help="Root directory of source GRIB files.",
    )
    p_prepare.add_argument(
        "--bundle-dir", default=None,
        help="Output directory for built bundles (default: <output-dir>/bundles).",
    )

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Run evaluators on existing predictions.")
    _add_common_args(p_eval)
    p_eval.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing prediction .nc files.",
    )
    _add_evaluator_filter_args(p_eval)
    _add_lane_override_args(p_eval)
    p_eval.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (for evaluators that require it, e.g. mechanistic).",
    )
    p_eval.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )
    p_eval.add_argument(
        "--run-label", default="",
        help="Short display label for this run (used in TC/plot legends). "
             "Overrides the automatic fallback from the predictions directory name.",
    )

    # --- scoreboard ---
    p_sb = subparsers.add_parser("scoreboard", help="Generate scoreboard from evaluation results.")
    _add_common_args(p_sb)
    p_sb.add_argument(
        "--eval-dir", required=True,
        help="Root evaluation directory containing evaluator outputs.",
    )
    _add_evaluator_filter_args(p_sb)

    return parser


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _parse_int_csv(raw: str) -> list[int]:
    """Parse comma-separated integers, sorted ascending."""
    return sorted(int(tok.strip()) for tok in raw.split(",") if tok.strip())


def _parse_str_csv(raw: str) -> list[str]:
    """Parse comma-separated strings, preserving order."""
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _build_lane_overrides(args: argparse.Namespace) -> dict:
    """Build overrides dict from CLI args that are not None."""
    predict_overrides: dict = {}
    if getattr(args, "members", None) is not None:
        predict_overrides["members"] = _parse_int_csv(args.members)
    if getattr(args, "steps", None) is not None:
        predict_overrides["steps"] = _parse_int_csv(args.steps)
    if getattr(args, "dates", None) is not None:
        predict_overrides["dates"] = _parse_str_csv(args.dates)
    if predict_overrides:
        return {"predict": predict_overrides}
    return {}


def _resolve_evaluators(args: argparse.Namespace, lane_config: dict) -> list[str]:
    """Three-step evaluator resolution.

    1. --only: run exactly those evaluators.
    2. --include-diagnostics: default + diagnostics groups.
    3. Otherwise: default group only.
    """
    evaluator_groups = lane_config.get("evaluator_groups", {})

    if getattr(args, "only", None) is not None:
        requested = _parse_str_csv(args.only)
        unknown = [e for e in requested if e not in ALL_EVALUATORS]
        if unknown:
            raise SystemExit(
                f"Unknown evaluator(s) in --only: {unknown}. "
                f"Valid evaluators: {ALL_EVALUATORS}"
            )
        return requested

    if getattr(args, "include_diagnostics", False):
        default_group = evaluator_groups.get("default", [])
        diag_group = evaluator_groups.get("diagnostics", [])
        # Preserve order, avoid duplicates
        combined: list[str] = list(default_group)
        for e in diag_group:
            if e not in combined:
                combined.append(e)
        return combined

    return list(evaluator_groups.get("default", []))


def _get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _resolve_output_dir(host_config: dict, lane_name: str) -> Path:
    """Build output directory: <scratch_root>/eval/<lane>/run_<YYYYMMDDTHHMMSS>/"""
    scratch_root = Path(host_config["scratch_root"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return scratch_root / "eval" / lane_name / f"run_{timestamp}"


def _config_file_paths(lane_name: str, host_name: str) -> dict:
    """Return paths to the YAML config files that were loaded."""
    config_dir = Path(__file__).resolve().parent / "config"
    return {
        "lane": str(config_dir / "lanes" / f"{lane_name}.yaml"),
        "host": str(config_dir / "hosts" / f"{host_name}.yaml"),
    }


def _build_effective_config(
    args: argparse.Namespace,
    lane_config: dict,
    host_config: dict,
    lane_name: str,
    host_name: str,
    overrides: dict,
    evaluators: list[str],
    output_dir: Path,
) -> dict:
    """Build the effective config dict for emission."""
    code_root = host_config.get("code_root", "unknown")
    return {
        "lane": lane_name,
        "host": host_name,
        "checkpoint": getattr(args, "checkpoint", None),
        "predictions_dir": getattr(args, "predictions_dir", None),
        "eval_dir": getattr(args, "eval_dir", None),
        "resolved": lane_config,
        "overrides": overrides,
        "cli_args": sys.argv[1:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "code_root": code_root,
        "config_file_paths": _config_file_paths(lane_name, host_name),
        "output_dir": str(output_dir),
        "evaluators": evaluators,
        "evaluators_run": [],
    }


def _write_effective_config(config: dict, output_dir: Path) -> Path:
    """Write effective_config.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "effective_config.json"
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    return path


def _update_effective_config_completion(
    output_dir: Path, evaluators_run: list[str],
) -> None:
    """Update effective_config.json with completion info."""
    path = output_dir / "effective_config.json"
    if path.exists():
        config = json.loads(path.read_text())
    else:
        config = {}
    config["evaluators_run"] = evaluators_run
    config["completion_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Run predictions via subprocess call to eval.predict.main."""
    predict_cfg = lane_config["predict"]
    checkpoint = args.checkpoint

    source_grib_root = getattr(args, "source_grib_root", None) or ""

    # --- Prepare: build truth-aware bundles if lane has prepare: section ---
    if lane_config.get("prepare") and source_grib_root:
        from eval.prepare.builder import build_bundles
        bundle_dir_arg = getattr(args, "bundle_dir", None)
        bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"
        bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
        if isinstance(bundle_pairs_raw, str):
            bundle_pairs_raw = [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]
        LOG.info("=== Phase 0: Bundle preparation ===")
        build_bundles(
            lane_config=lane_config,
            bundle_dir=bundle_dir,
            source_grib_root=source_grib_root,
            dates=list(predict_cfg.get("dates", [])),
            steps=[int(s) for s in predict_cfg.get("steps", [])],
            members=[int(m) for m in predict_cfg.get("members", [])],
            bundle_pairs=list(bundle_pairs_raw),
            verification_path=output_dir / "bundle_build_verification.json",
        )
        input_root = str(bundle_dir)
    else:
        # Resolve input_root: lane config takes precedence over host DATA_DIR
        input_root = predict_cfg.get("input_root", "")
        if not input_root:
            env_setup = host_config.get("environment_setup", {})
            exports = env_setup.get("exports", {})
            input_root = exports.get("DATA_DIR", "")

    members_str = ",".join(str(m) for m in predict_cfg["members"])
    steps_str = ",".join(str(s) for s in predict_cfg["steps"])
    dates_str = ",".join(predict_cfg["dates"])
    bundle_pairs = predict_cfg.get("bundle_pairs", "")
    if isinstance(bundle_pairs, list):
        bundle_pairs = ",".join(
            f"{item.get('date')}:{item.get('step')}" if isinstance(item, dict) else str(item)
            for item in bundle_pairs
        )

    predictions_dir = output_dir / "predictions"

    cmd = [
        sys.executable, "-m", "eval.predict.main",
        "--name-ckpt", str(checkpoint),
        "--out-dir", str(predictions_dir),
        "--members", members_str,
        "--steps", steps_str,
        "--dates", dates_str,
        "--input-root", input_root,
        "--allow-existing-out-dir",
    ]
    if bundle_pairs:
        cmd += ["--bundle-pairs", str(bundle_pairs)]

    LOG.info("Running predictions: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    LOG.info("Predictions written to %s", predictions_dir)


def _run_evaluators(
    predictions_dir: Path,
    lane_config: dict,
    evaluators: list[str],
    output_dir: Path,
    *,
    overwrite: bool = False,
    checkpoint: str | None = None,
    run_label: str = "",
) -> list[str]:
    """Run selected evaluators on existing predictions. Returns list of evaluators that ran."""
    evaluators_run: list[str] = []

    for name in evaluators:
        if name not in ALL_EVALUATORS:
            LOG.warning(
                "Skipping unknown evaluator '%s'. Valid: %s", name, ALL_EVALUATORS
            )
            continue

        # Import evaluator module
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError:
            LOG.error(
                "Cannot import evaluator 'eval.evaluators.%s'. "
                "Check that the module exists and has no import errors.",
                name,
            )
            continue

        spec = getattr(mod, "EVALUATOR_SPEC", {})

        # Check requirements
        requires = spec.get("requires", [])
        if "checkpoint" in requires and not checkpoint:
            LOG.warning(
                "Skipping evaluator '%s': requires 'checkpoint' but none provided. "
                "Pass --checkpoint to include it.",
                name,
            )
            continue

        # Determine results directory
        results_dir = output_dir / "evaluators" / name

        # Check overwrite guard
        if results_dir.exists() and not overwrite:
            LOG.warning(
                "Evaluator '%s' output already exists at %s. "
                "Use --overwrite to re-run. Skipping.",
                name, results_dir,
            )
            continue

        if results_dir.exists() and overwrite:
            import shutil
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        eval_config = lane_config.get(name, {})

        LOG.info("Running evaluator: %s", name)

        # Run
        run_fn = getattr(mod, "run", None)
        if run_fn is not None:
            try:
                run_fn(
                    predictions_dir, lane_config, eval_config,
                    output_dir=results_dir, overwrite=overwrite,
                    checkpoint=checkpoint,
                    run_label=run_label,
                )
            except Exception:
                LOG.error("Evaluator '%s' run() failed", name, exc_info=True)
                continue

        # Score
        score_fn = getattr(mod, "score", None)
        if score_fn is not None:
            try:
                scores = score_fn(
                    results_dir, lane_config, eval_config,
                    predictions_dir=predictions_dir,
                )
                if scores:
                    metrics_path = results_dir / "metrics.json"
                    metrics_path.write_text(
                        json.dumps(scores, indent=2, default=str) + "\n"
                    )
            except Exception:
                LOG.error("Evaluator '%s' score() failed", name, exc_info=True)

        # Plot
        plot_fn = getattr(mod, "plot", None)
        if plot_fn is not None:
            try:
                plot_fn(results_dir, lane_config, eval_config, output_dir=results_dir)
            except Exception:
                LOG.error("Evaluator '%s' plot() failed", name, exc_info=True)

        evaluators_run.append(name)
        LOG.info("Evaluator '%s' completed. Output: %s", name, results_dir)

    return evaluators_run


def _resolve_run_root(output_dir: Path) -> Path:
    """Resolve the run root from output_dir.

    output_dir must be either the run root itself (containing evaluators/)
    directly) or <run_root>/data (the canonical layout when evaluate is
    called with --predictions-dir).  Any other structure raises ValueError
    so callers cannot silently misplace outputs.
    """
    if output_dir.name == "data":
        return output_dir.parent
    if (output_dir / "evaluators").exists() or (output_dir / "predictions").exists():
        return output_dir
    raise ValueError(
        f"Cannot resolve run root from output_dir={output_dir!r}. "
        "Expected either <run_root>/data or a directory containing evaluators/ or predictions/."
    )


def _consolidate_plots(output_dir: Path) -> None:
    """Copy all PDFs and PNGs from evaluators/* to <run_root>/plots/.

    Always writes to <run_root>/plots/ regardless of how output_dir is
    structured, so plots are never buried inside data/.
    """
    import shutil
    run_root = _resolve_run_root(output_dir)
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    evaluators_dir = output_dir / "evaluators"
    if not evaluators_dir.exists():
        return
    for src in sorted(evaluators_dir.rglob("*.pdf")) + sorted(evaluators_dir.rglob("*.png")):  # type: ignore[operator]
        shutil.copy2(src, plots_dir / src.name)
    LOG.info("Plots consolidated to %s (%d files)", plots_dir, len(list(plots_dir.iterdir())))


def _run_scoreboard(
    eval_dir: Path,
    lane_config: dict,
    evaluators: list[str],
    output_dir: Path,
) -> None:
    """Generate scoreboard from evaluation results."""
    from eval.scoreboard.aggregator import aggregate_scores
    from eval.scoreboard.formatter import to_csv, to_markdown, to_pretty_text

    scores = aggregate_scores(eval_dir, lane_config, evaluators=evaluators)

    # Write outputs
    scoreboard_dir = output_dir / "scoreboard"
    scoreboard_dir.mkdir(parents=True, exist_ok=True)

    csv_path = to_csv(scores, scoreboard_dir / "scores.csv")
    md_path = to_markdown(scores, scoreboard_dir / "scores.md")
    text = to_pretty_text(scores)

    LOG.info("Scoreboard CSV:      %s", csv_path)
    LOG.info("Scoreboard Markdown: %s", md_path)
    print("\n--- Scoreboard ---")
    print(text)
    print()


def cmd_prepare(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Build truth-aware bundles only (no prediction)."""
    from eval.prepare.builder import build_bundles

    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise SystemExit(f"Lane '{args.lane}' has no 'prepare:' section in its config.")

    source_grib_root = args.source_grib_root
    bundle_dir_arg = getattr(args, "bundle_dir", None)
    bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"

    predict_cfg = lane_config.get("predict", {})
    bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
    if isinstance(bundle_pairs_raw, str):
        bundle_pairs_raw = [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]

    build_bundles(
        lane_config=lane_config,
        bundle_dir=bundle_dir,
        source_grib_root=source_grib_root,
        dates=list(predict_cfg.get("dates", [])),
        steps=[int(s) for s in predict_cfg.get("steps", [])],
        members=[int(m) for m in predict_cfg.get("members", [])],
        bundle_pairs=list(bundle_pairs_raw),
        verification_path=bundle_dir.parent / "bundle_build_verification.json",
    )
    LOG.info("Bundle preparation complete. Bundles in: %s", bundle_dir)


def cmd_run(args: argparse.Namespace, lane_config: dict, host_config: dict, evaluators: list[str], output_dir: Path) -> None:
    """Full pipeline: predict + evaluate + scoreboard."""
    predictions_dir = output_dir / "predictions"
    checkpoint = args.checkpoint

    # Step 1: Predict
    LOG.info("=== Phase 1/3: Predictions ===")
    cmd_predict(args, lane_config, host_config, output_dir)

    # Step 2: Evaluate
    LOG.info("=== Phase 2/3: Evaluators ===")
    evaluators_run = _run_evaluators(
        predictions_dir, lane_config, evaluators, output_dir,
        overwrite=getattr(args, "overwrite", False),
        checkpoint=checkpoint,
    )

    # Step 3: Scoreboard
    LOG.info("=== Phase 3/3: Scoreboard ===")
    _run_scoreboard(output_dir, lane_config, evaluators, output_dir)

    # Step 4: Consolidate all plots to <run_root>/plots/
    _consolidate_plots(output_dir)

    # Update effective config with completion info
    _update_effective_config_completion(output_dir, evaluators_run)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse args, resolve config, dispatch to subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # --- Resolve config ---
    lane_name = args.lane
    host_name = args.host or DEFAULT_HOST

    lane_overrides = _build_lane_overrides(args)

    try:
        lane_config = load_lane(lane_name, overrides=lane_overrides or None)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Lane config not found: '{lane_name}'. "
            f"Available lanes are YAML files in eval/config/lanes/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load lane config '{lane_name}': {exc}") from exc

    try:
        host_config = load_host(host_name)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Host config not found: '{host_name}'. "
            f"Available hosts are YAML files in eval/config/hosts/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load host config '{host_name}': {exc}") from exc

    # --- Resolve evaluators (for subcommands that need them) ---
    evaluators: list[str] = []
    if args.subcommand in ("run", "evaluate", "scoreboard"):
        evaluators = _resolve_evaluators(args, lane_config)

    # --- Resolve output dir ---
    if args.subcommand == "scoreboard" and hasattr(args, "eval_dir") and args.eval_dir:
        output_dir = Path(args.eval_dir)
    elif args.subcommand == "evaluate" and hasattr(args, "predictions_dir") and args.predictions_dir:
        # Place evaluator outputs alongside predictions
        output_dir = Path(args.predictions_dir).parent
    elif args.subcommand == "prepare":
        bundle_dir_arg = getattr(args, "bundle_dir", None)
        output_dir = Path(bundle_dir_arg).parent if bundle_dir_arg else _resolve_output_dir(host_config, lane_name)
    else:
        output_dir = _resolve_output_dir(host_config, lane_name)

    # --- Build effective config ---
    effective = _build_effective_config(
        args, lane_config, host_config,
        lane_name, host_name, lane_overrides,
        evaluators, output_dir,
    )

    # --- Dry run ---
    if args.dry_run:
        print(json.dumps(effective, indent=2, default=str))
        return

    # --- Preflight: write effective config ---
    config_path = _write_effective_config(effective, output_dir)
    LOG.info("Effective config written to %s", config_path)

    # --- Validate evaluator names ---
    unknown_evals = [e for e in evaluators if e not in ALL_EVALUATORS]
    if unknown_evals:
        raise SystemExit(
            f"Unknown evaluator(s) in lane config evaluator_groups: {unknown_evals}. "
            f"Valid evaluators: {ALL_EVALUATORS}"
        )

    # --- Dispatch ---
    if args.subcommand == "run":
        cmd_run(args, lane_config, host_config, evaluators, output_dir)
    elif args.subcommand == "predict":
        cmd_predict(args, lane_config, host_config, output_dir)
    elif args.subcommand == "prepare":
        cmd_prepare(args, lane_config, host_config, output_dir)
    elif args.subcommand == "evaluate":
        predictions_dir = Path(args.predictions_dir)
        evaluators_run = _run_evaluators(
            predictions_dir, lane_config, evaluators, output_dir,
            overwrite=getattr(args, "overwrite", False),
            checkpoint=getattr(args, "checkpoint", None),
            run_label=getattr(args, "run_label", ""),
        )
        _consolidate_plots(output_dir)
        _update_effective_config_completion(output_dir, evaluators_run)
    elif args.subcommand == "scoreboard":
        eval_dir = Path(args.eval_dir)
        _run_scoreboard(eval_dir, lane_config, evaluators, output_dir)


if __name__ == "__main__":
    main()
