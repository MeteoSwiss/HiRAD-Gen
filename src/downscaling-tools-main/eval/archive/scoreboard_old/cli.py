"""CLI for scoreboard scoring — score-tc, score-spectra, score-surface, verify."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _score_tc(args: argparse.Namespace) -> None:
    from eval._backends.scoreboard.canonical_data import load_canonical_analysis
    from eval._backends.scoreboard.tc import load_tc_extreme_scores_from_json

    event_names = tuple(args.events.split(",")) if args.events else None
    canonical = None
    if args.lane:
        canonical = load_canonical_analysis(args.lane)

    result = load_tc_extreme_scores_from_json(
        Path(args.stats_json),
        run_id=args.run_id,
        event_names=event_names,
        canonical_analysis_by_event=canonical,
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _score_spectra(args: argparse.Namespace) -> None:
    from eval._backends.scoreboard.spectra import load_spectra_metrics

    ref_root = Path(args.reference_root) if args.reference_root else None
    result = load_spectra_metrics(Path(args.spectra_dir), reference_root=ref_root)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _score_surface(args: argparse.Namespace) -> None:
    from eval._backends.scoreboard.surface import load_surface_loss_metrics

    result = load_surface_loss_metrics(Path(args.summary_json))
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _verify(args: argparse.Namespace) -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).parent / "tests"), "-v"],
        capture_output=False,
    )
    sys.exit(result.returncode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eval.scoreboard",
        description="Scoreboard scoring library CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tc_parser = sub.add_parser("score-tc", help="Score TC extremes from stats JSON")
    tc_parser.add_argument("--stats-json", required=True, help="Path to TC stats JSON")
    tc_parser.add_argument("--run-id", required=True, help="Run ID to match")
    tc_parser.add_argument("--events", default=None, help="Comma-separated event names")
    tc_parser.add_argument("--lane", default=None, help="Resolution lane (e.g. o320)")
    tc_parser.set_defaults(func=_score_tc)

    spectra_parser = sub.add_parser("score-spectra", help="Score spectra L2 distances")
    spectra_parser.add_argument("--spectra-dir", required=True, help="Path to spectra directory")
    spectra_parser.add_argument("--reference-root", default=None, help="Reference root path")
    spectra_parser.set_defaults(func=_score_spectra)

    surface_parser = sub.add_parser("score-surface", help="Score surface loss metrics")
    surface_parser.add_argument("--summary-json", required=True, help="Path to surface loss JSON")
    surface_parser.set_defaults(func=_score_surface)

    verify_parser = sub.add_parser("verify", help="Run golden-file verification tests")
    verify_parser.set_defaults(func=_verify)

    args = parser.parse_args(argv)
    args.func(args)
