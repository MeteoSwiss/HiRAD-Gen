#!/usr/bin/env python3
"""Generate the ENFO O320 scoreboard from evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

from eval.paths import DEFAULT_EVAL_ROOT, default_scoreboard_path
from eval._backends.scoreboard.surface import surface_variable_nmse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXT_SCOREBOARD_CSV = Path(
    "/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/docs/docs/scoreboard_o96_o320/state/source_26_30/scoreboard.csv"
)
EEFO_O96_LABEL = "eefo_o96"
EEFO_O96_SHORT_ID = "x_interp"
EXPECTED_CONTEXT_LANE = "o96_o320"
EXPECTED_CONTEXT_DATES = (20230826, 20230827, 20230828, 20230829, 20230830)
EXPECTED_CONTEXT_STEPS = (24, 48, 72, 96, 120)
SURFACE_NMSE_COLUMNS = (
    ("surface_10v", "10v", "10v nMSE"),
    ("surface_2t", "2t", "2t nMSE"),
    ("surface_msl", "msl", "MSLP nMSE"),
    ("surface_sp", "sp", "SP nMSE"),
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.jobs import scoreboard_metrics as metrics


def _fmt(x: float, precision: int = 4) -> str:
    """Format a float for display, showing 'na' for non-finite values."""
    if not math.isfinite(x):
        return "na"
    if precision == 3:
        return f"{x:.3f}"
    elif precision == 4:
        return f"{x:.4f}"
    elif precision == 6:
        return f"{x:.6e}"
    else:
        return f"{x:.{precision}f}"


def load_sigma_results(eval_root: Path) -> dict[str, dict[str, float]]:
    """Load sigma evaluation CSV files keyed by the emitting run identifier."""
    results: dict[str, dict[str, float]] = {}
    sigma_dir = eval_root / "scoreboards" / "sigma"
    if not sigma_dir.exists():
        return results

    for csv_file in sorted(sigma_dir.glob("*_sigma_eval.csv")):
        run_id = csv_file.stem.replace("_sigma_eval", "")
        sigma_losses = metrics.load_sigma_losses_from_csv(csv_file)
        if sigma_losses:
            results[run_id] = sigma_losses
    return results


def load_tc_results(eval_root: Path) -> dict[str, dict[str, Any]]:
    """Load TC stats with the same fallback used by the docs-side scoreboard flow."""
    results: dict[str, dict[str, Any]] = {}

    def tc_candidate_rank(path: Path) -> tuple[int, int, int, int, str]:
        text = str(path).lower()
        name = path.name.lower()
        return (
            1 if "proxy" in text else 0,
            1 if any(token in text for token in ("smoketest", "cached_refs", "template", "submit_helper")) else 0,
            1 if any(token in text for token in ("full75", "aug16_30")) else 0,
            0 if any(token in name for token in ("from_predictions", "full25")) else 1,
            0 if "idalia_franklin" in name else 1,
            text,
        )

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        if not (run_root / "surface_loss_summary.json").exists():
            continue
        tc_scores: dict[str, float] = {}
        candidates = sorted(run_root.rglob("tc_normed_pdfs_*.stats.json"), key=tc_candidate_rank)
        for stats_file in candidates:
            parsed = metrics.load_tc_extreme_scores_from_json(stats_file, run_id=run_root.name)
            for event_name, score in parsed.items():
                tc_scores.setdefault(event_name, score)
            if {"idalia", "franklin"} <= tc_scores.keys():
                break
        if tc_scores:
            results[run_root.name] = {
                (f"{k}_extreme" if not k.endswith("_enfo_dev") else k): v
                for k, v in tc_scores.items()
            }
    return results


def load_spectra_results(eval_root: Path) -> dict[str, float]:
    """Load full25 spectra metrics using the 3-field scoreboard contract."""
    results: dict[str, float] = {}

    def spectra_candidate_rank(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        return (
            0 if "proxy10_subset" in name else 1,
            0 if "ecmwf" in name else 1,
            name,
        )

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        if not (run_root / "surface_loss_summary.json").exists():
            continue
        spectra_dirs = sorted(
            (path for path in run_root.glob("spectra*") if path.is_dir()),
            key=spectra_candidate_rank,
        )
        for spectra_dir in spectra_dirs:
            try:
                spectra_data = metrics.load_spectra_metrics(spectra_dir)
            except ValueError as exc:
                print(f"Skipping incompatible spectra dir {spectra_dir}: {exc}")
                continue
            mean_l2 = spectra_data.get("mean")
            if mean_l2 is None:
                continue
            results[run_root.name] = float(mean_l2)
            break
    return results


def load_surface_loss_results(
    eval_root: Path,
    *,
    truth_std_by_variable: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load surface loss JSON files."""
    results: dict[str, dict[str, Any]] = {}

    for summary_file in sorted(eval_root.glob("*/surface_loss_summary.json")):
        run_id = summary_file.parent.name
        surface_loss = metrics.load_surface_loss_metrics(
            summary_file,
            truth_std_by_variable=truth_std_by_variable,
        )
        if surface_loss.get("weighted_mse") is not None:
            results[run_id] = surface_loss
    return results


def load_inference_results(eval_root: Path) -> dict[str, str]:
    """Load short inference labels from run-root metadata or logs."""
    results: dict[str, str] = {}

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        if not (run_root / "surface_loss_summary.json").exists():
            continue
        results[run_root.name] = metrics.infer_eval_sampler_min_from_run_root(run_root)
    return results


def _nan() -> float:
    return float("nan")


def _csv_metric(row: dict[str, str], key: str) -> float:
    value = metrics.finite_float(row.get(key))
    return value if value is not None else _nan()



def _surface_column_defaults() -> dict[str, float]:
    return {column_key: _nan() for column_key, _, _ in SURFACE_NMSE_COLUMNS}


def _apply_surface_columns(row: dict[str, Any], surface_metrics: dict[str, Any]) -> None:
    for column_key, variable, _ in SURFACE_NMSE_COLUMNS:
        value = surface_variable_nmse(surface_metrics, variable)
        if value is not None and math.isfinite(value):
            row[column_key] = float(value)


def _bundle_scope_signature(config: dict[str, Any]) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    lane = str(config.get("lane") or "").strip()
    source = config.get("source")
    scope = source.get("bundle_scope") if isinstance(source, dict) else {}
    if not isinstance(scope, dict):
        scope = {}

    def _ints(raw: Any) -> tuple[int, ...]:
        values: list[int] = []
        if isinstance(raw, (list, tuple)):
            for value in raw:
                number = metrics.finite_float(value)
                if number is None:
                    continue
                values.append(int(number))
        return tuple(values)

    dates = _ints(scope.get("dates"))
    steps = _ints(scope.get("steps_hours"))
    return lane, dates, steps


def _predictions_support_xinterp(predictions_dir: Path) -> bool:
    prediction_files = sorted(predictions_dir.glob("predictions_*.nc"))
    if not prediction_files:
        return False
    try:
        import xarray as xr

        with xr.open_dataset(prediction_files[0]) as ds:
            return "x_interp" in ds.variables and "y" in ds.variables
    except Exception:
        return False


def choose_xinterp_context_predictions_dir(eval_root: Path) -> Path | None:
    candidates: list[tuple[tuple[Any, ...], Path]] = []

    for run_root in sorted(path for path in eval_root.glob("manual_*") if path.is_dir()):
        predictions_dir = run_root / "predictions"
        prediction_files = sorted(predictions_dir.glob("predictions_*.nc")) if predictions_dir.exists() else []
        if not prediction_files:
            continue
        if not _predictions_support_xinterp(predictions_dir):
            continue
        config = metrics.load_mapping_file(run_root / "EXPERIMENT_CONFIG.yaml")
        lane, dates, steps = _bundle_scope_signature(config)
        exact_contract = (
            lane == EXPECTED_CONTEXT_LANE
            and dates == EXPECTED_CONTEXT_DATES
            and steps == EXPECTED_CONTEXT_STEPS
        )
        rank = (
            0 if exact_contract else 1,
            0 if lane == EXPECTED_CONTEXT_LANE else 1,
            0 if dates == EXPECTED_CONTEXT_DATES else 1,
            0 if steps == EXPECTED_CONTEXT_STEPS else 1,
            -len(prediction_files),
            _run_rank(run_root.name),
        )
        candidates.append((rank, predictions_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def load_context_baseline_rows(
    source_csv: Path,
    *,
    context_surface_metrics: dict[str, dict[str, Any]] | None = None,
    labels: tuple[str, ...] = (EEFO_O96_LABEL,),
) -> list[dict[str, Any]]:
    """Load curated non-run baseline rows from the docs-side source scoreboard CSV."""
    if not source_csv.exists():
        return []

    context_surface_metrics = context_surface_metrics or {}

    wanted = set(labels)
    rows: list[dict[str, Any]] = []
    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            label = str(raw_row.get("label") or "").strip()
            table_group = str(raw_row.get("table_group") or "").strip()
            if table_group != "context" or label not in wanted:
                continue
            surface_loss = _csv_metric(raw_row, "surface_weighted_mse")
            surface_loss_text = ""
            if label in context_surface_metrics:
                context_surface = context_surface_metrics[label]
                context_surface_value = metrics.finite_float(context_surface.get("weighted_nmse"))
                if context_surface_value is not None:
                    surface_loss = context_surface_value
                    surface_loss_text = metrics.format_surface_loss_for_scoreboard(context_surface)
            rows.append(
                {
                    "row_key": label,
                    "display_run_id": label,
                    "short_id": EEFO_O96_SHORT_ID if label == EEFO_O96_LABEL else label,
                    "inference": "na",
                    "sigma_1": _nan(),
                    "sigma_5": _nan(),
                    "sigma_10": _nan(),
                    "sigma_100": _nan(),
                    "idalia_extreme": _csv_metric(raw_row, "idalia_tc_extreme_score"),
                    "franklin_extreme": _csv_metric(raw_row, "franklin_tc_extreme_score"),
                    "enfo_deviation": _csv_metric(raw_row, "enfo_deviation"),
                    "spectra_l2": _csv_metric(raw_row, "spectra_mean_distance"),
                    "surface_loss": surface_loss,
                    "surface_loss_text": surface_loss_text,
                    "row_group": "context",
                    **_surface_column_defaults(),
                }
            )
            if label in context_surface_metrics:
                _apply_surface_columns(rows[-1], context_surface_metrics[label])
    rows.sort(key=lambda row: row["display_run_id"])
    return rows


def _row_token(run_id: str) -> str:
    return metrics.extract_checkpoint_token(run_id) or run_id


def _row_key_map(run_ids: set[str]) -> dict[str, str]:
    tokens = {
        token
        for run_id in run_ids
        if run_id and (token := metrics.extract_checkpoint_token(run_id))
    }
    canonical_token: dict[str, str] = {}
    for token in tokens:
        related = [other for other in tokens if other.startswith(token) or token.startswith(other)]
        canonical_token[token] = max(related, key=len) if related else token

    mapped: dict[str, str] = {}
    for run_id in run_ids:
        token = metrics.extract_checkpoint_token(run_id)
        mapped[run_id] = canonical_token.get(token, run_id) if token else run_id
    return mapped


def _run_rank(run_id: str) -> tuple[int, int, int, int, str]:
    return (
        0 if run_id.startswith("manual_") else 1,
        1 if "proxy" in run_id else 0,
        0 if "new_o96_o320" in run_id else 1,
        -len(run_id),
        run_id,
    )


def _prefer_run_id(existing: str | None, candidate: str) -> str:
    if existing is None:
        return candidate
    return candidate if _run_rank(candidate) < _run_rank(existing) else existing


def build_scoreboard_rows(
    sigma_data: dict[str, dict[str, float]],
    tc_data: dict[str, dict[str, Any]],
    spectra_data: dict[str, float],
    surface_loss_data: dict[str, Any],
    inference_data: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    all_run_ids = set(sigma_data) | set(tc_data) | set(spectra_data) | set(surface_loss_data)
    if inference_data:
        all_run_ids |= set(inference_data)
    row_key_map = _row_key_map(all_run_ids)

    rows_by_key: dict[str, dict[str, Any]] = {}

    def ensure_row(run_id: str) -> dict[str, Any]:
        row_key = row_key_map[run_id]
        row = rows_by_key.get(row_key)
        if row is None:
            row = {
                "row_key": row_key,
                "display_run_id": None,
                "inference": "na",
                "row_group": "experiment",
                "inference_source_run_id": None,
                "sigma_1": _nan(),
                "sigma_5": _nan(),
                "sigma_10": _nan(),
                "sigma_100": _nan(),
                "idalia_extreme": _nan(),
                "franklin_extreme": _nan(),
                "enfo_deviation": _nan(),
                "spectra_l2": _nan(),
                "surface_loss": _nan(),
                "surface_loss_text": "",
                **_surface_column_defaults(),
            }
            rows_by_key[row_key] = row
        row["display_run_id"] = _prefer_run_id(row["display_run_id"], run_id)
        return row

    for run_id, sigma in sigma_data.items():
        row = ensure_row(run_id)
        for key in ("sigma_1", "sigma_5", "sigma_10", "sigma_100"):
            value = sigma.get(key)
            if value is not None and math.isfinite(value):
                row[key] = float(value)

    for run_id, tc in tc_data.items():
        row = ensure_row(run_id)
        for key in ("idalia_extreme", "franklin_extreme"):
            value = tc.get(key)
            if value is not None and math.isfinite(value):
                row[key] = float(value)
        # Aggregate ENFO deviation across events
        enfo_devs = [
            v for k, v in tc.items()
            if k.endswith("_enfo_dev") and v is not None and math.isfinite(v)
        ]
        if enfo_devs:
            row["enfo_deviation"] = sum(enfo_devs) / len(enfo_devs)

    for run_id, spectra_l2 in spectra_data.items():
        row = ensure_row(run_id)
        if math.isfinite(spectra_l2):
            row["spectra_l2"] = float(spectra_l2)

    for run_id, surface_loss in surface_loss_data.items():
        row = ensure_row(run_id)
        surface_value = metrics.finite_float(surface_loss)
        surface_text = ""
        if isinstance(surface_loss, dict):
            surface_value = metrics.finite_float(
                surface_loss.get(
                    "weighted_nmse",
                    surface_loss.get("weighted_surface_nmse", surface_loss.get("weighted_mse")),
                )
            )
            surface_text = metrics.format_surface_loss_for_scoreboard(surface_loss)
        if surface_value is not None:
            row["surface_loss"] = float(surface_value)
        if surface_text and surface_text != "na":
            row["surface_loss_text"] = surface_text
        if isinstance(surface_loss, dict):
            _apply_surface_columns(row, surface_loss)

    for run_id, inference in (inference_data or {}).items():
        text = str(inference or "").strip() or "na"
        if text == "na":
            continue
        row = ensure_row(run_id)
        chosen = _prefer_run_id(row["inference_source_run_id"], run_id)
        if chosen == run_id:
            row["inference"] = text
            row["inference_source_run_id"] = run_id

    rows = sorted(rows_by_key.values(), key=lambda row: row["display_run_id"])
    for row in rows:
        token = metrics.extract_checkpoint_token(row["row_key"]) or metrics.extract_checkpoint_token(row["display_run_id"] or "")
        row["short_id"] = token[:8] if token else (row["display_run_id"] or row["row_key"])[:8]
        row["display_run_id"] = row["display_run_id"] or row["row_key"]
        row.pop("inference_source_run_id", None)
    return rows


def generate_scoreboard_markdown(rows: list[dict[str, Any]]) -> str:
    """Generate markdown table from normalized scoreboard rows."""
    lines = [
        "# ENFO O320 Scoreboard",
        "",
        "Standard evaluation protocol for o96→o320 downscaling runs.",
        "Lower is better for sigma, spectra, and surface loss; higher is better for TC extreme scores.",
        "",
        "| Ckpt | Inference | Run ID | σ=1 loss | σ=5 loss | σ=10 loss | σ=100 loss | TC Idalia | TC Franklin | ENFO dev | Spectra L2 | Sfc nMSE | 10v nMSE | 2t nMSE | MSLP nMSE | SP nMSE |",
        "|------|-----------|--------|----------|----------|-----------|------------|----------|-------------|----------|------------|----------|----------|----------|-----------|---------|",
    ]

    for row in rows:
        surface_loss_text = str(row.get("surface_loss_text", "")).strip() or _fmt(row["surface_loss"], 4)
        line = (
            f"| {row['short_id']} | {row.get('inference', 'na')} | {row['display_run_id']} | "
            f"{_fmt(row['sigma_1'])} | {_fmt(row['sigma_5'])} | "
            f"{_fmt(row['sigma_10'])} | {_fmt(row['sigma_100'])} | "
            f"{_fmt(row['idalia_extreme'], 3)} | {_fmt(row['franklin_extreme'], 3)} | "
            f"{_fmt(row['enfo_deviation'], 3)} | "
            f"{_fmt(row['spectra_l2'], 4)} | {surface_loss_text} | "
            f"{_fmt(row['surface_10v'], 4)} | {_fmt(row['surface_2t'], 4)} | "
            f"{_fmt(row['surface_msl'], 4)} | {_fmt(row['surface_sp'], 4)} |"
        )
        lines.append(line)

    lines.extend([
        "",
        "---",
        "",
        "**Legend:**",
        "- **Context baseline rows**: Curated comparison rows appended after experiment runs; `x_interp` is sourced from the docs-side `eefo_o96` input baseline",
        "- **Inference**: Schedule-plus-step label inferred from run metadata or prediction logs (for example `piecewise30`, `karras40`)",
        "- **Sigma loss**: Diffusion validation loss at fixed noise levels (σ=1,5,10,100)",
        "- **TC Idalia/Franklin**: Analysis-anchored TC extreme score (0-1, higher=closer to analysis truth)",
        "- **ENFO dev**: ENFO deviation — distance from ENFO baseline (context metric, not a penalty)",
        "- **Spectra L2**: Fine-scale-weighted mean relative L2 across 6 weather variables (10u, 10v, 2t, msl, t_850, z_500)",
        "- **Sfc nMSE**: Area-weighted and variable-weighted surface MSE after per-variable truth-std normalization over the fixed Aug 26-30 evaluation contract",
        "- **10v / 2t / MSLP / SP nMSE**: Per-variable truth-std-normalized surface MSE for the named field, using the same fixed evaluation contract as the aggregate surface score",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate enfo_o320_scoreboard.md from evaluation artifacts"
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default=DEFAULT_EVAL_ROOT,
        help="Root directory containing evaluation outputs",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=default_scoreboard_path("enfo_o320_scoreboard.md"),
        help="Output markdown file path",
    )
    parser.add_argument(
        "--context-csv",
        type=str,
        default=str(DEFAULT_CONTEXT_SCOREBOARD_CSV),
        help="Docs-side source scoreboard CSV used for curated context baseline rows",
    )
    
    args = parser.parse_args()
    
    eval_root = Path(args.eval_root)
    out_md = Path(args.out_md)
    context_csv = Path(args.context_csv)
    
    print(f"Loading evaluation data from: {eval_root}")

    context_surface_metrics: dict[str, dict[str, Any]] = {}
    truth_std_by_variable: dict[str, float] | None = None
    context_predictions_dir = choose_xinterp_context_predictions_dir(eval_root)
    if context_predictions_dir is not None:
        try:
            eefo_surface = metrics.load_x_interp_surface_metrics(context_predictions_dir)
        except ValueError:
            pass
        else:
            context_surface_metrics[EEFO_O96_LABEL] = eefo_surface
            truth_std_by_variable = dict(eefo_surface.get("truth_std_by_variable", {}))

    sigma_data = load_sigma_results(eval_root)
    print(f"Loaded sigma data for {len(sigma_data)} runs")

    tc_data = load_tc_results(eval_root)
    print(f"Loaded TC data for {len(tc_data)} runs")

    spectra_data = load_spectra_results(eval_root)
    print(f"Loaded spectra data for {len(spectra_data)} runs")

    surface_loss_data = load_surface_loss_results(
        eval_root,
        truth_std_by_variable=truth_std_by_variable,
    )
    print(f"Loaded surface loss data for {len(surface_loss_data)} runs")

    inference_data = load_inference_results(eval_root)
    print(f"Loaded inference data for {len(inference_data)} runs")

    context_rows = load_context_baseline_rows(
        context_csv,
        context_surface_metrics=context_surface_metrics,
    )
    print(f"Loaded {len(context_rows)} curated context baseline rows")

    rows = build_scoreboard_rows(
        sigma_data,
        tc_data,
        spectra_data,
        surface_loss_data,
        inference_data=inference_data,
    )
    rows.extend(context_rows)
    markdown = generate_scoreboard_markdown(rows)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w") as f:
        f.write(markdown)

    print(f"\nScoreboard written to: {out_md}")


if __name__ == "__main__":
    main()
