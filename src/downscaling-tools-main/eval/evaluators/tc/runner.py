"""TC evaluator — data loading and per-event statistics.

Delegates to eval.tc.workflows for the heavy lifting, but sources
configuration from lane_config instead of hardcoded values, and uses
eval.discovery.predictions for file finding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.config.loader import load_event
from eval.discovery.predictions import find_predictions
from eval._backends.tc.data_types import BoundingBox
from eval._backends.tc.events import EVENTS, TCEvent
from eval._backends.tc.experiment_config import TCExperimentConfig
from eval._backends.tc.loading_predictions import (
    event_days_steps,
    forecast_dates_for_event,
    load_prediction_curves,
    select_prediction_files_for_event,
)
from eval._backends.tc.plot_config import PLOT_CONFIGS, TCPlotConfig
from eval._backends.tc.workflows import (
    _json_default,
    compute_event_stats,
    load_curves_for_event,
    run_member_maps,
)

LOG = logging.getLogger(__name__)


def _event_from_config(event_name: str) -> TCEvent:
    """Resolve a TCEvent, checking both the YAML config and the legacy registry."""
    if event_name in EVENTS:
        return EVENTS[event_name]
    cfg = load_event(event_name)
    return TCEvent(
        name=cfg["name"],
        year=str(cfg["dates"][0])[:4],
        month=str(cfg["dates"][0])[4:6],
        dates=[str(d)[6:8] for d in cfg["dates"]],
        analysis_dates=cfg.get("analysis_dates", [str(cfg["dates"][0])]),
        bbox=BoundingBox(
            north=cfg["lat_max"],
            south=cfg["lat_min"],
            east=cfg["lon_max"],
            west=cfg["lon_min"],
        ),
    )


def _pred_files_as_tuples(predictions_dir: Path):
    """Convert discovery PredictionFile objects to the (path, ymd, step) tuples
    that the existing TC loading code expects."""
    preds = find_predictions(predictions_dir)
    return [(p.path, int(p.date), p.step) for p in preds]


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    grib_dir: str | None = None,
    analysis_expid: str | None = None,
    support_mode: str = "native",
    extra_grib_references: dict | None = None,
    **kwargs,
) -> Path:
    """Run TC evaluation for all configured events.

    Writes per-event stats JSON to output_dir/stats.json.
    Returns the output directory path.
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "tc"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"TC output directory already has content: {output_dir}. Pass overwrite=True to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fall back to lane config for reference data not supplied by the caller.
    grib_dir = grib_dir or eval_config.get("grib_dir")
    analysis_expid = analysis_expid or eval_config.get("analysis_expid")
    reference_expids: tuple[str, ...] = tuple(eval_config.get("reference_expids") or ())
    support_mode = eval_config.get("support_mode", support_mode)

    # run_label: explicit arg > eval_config > grandparent-of-predictions (handles data/ lean layout)
    if not run_label:
        run_label = eval_config.get("run_label", "")
    if not run_label:
        parent = predictions_dir.parent
        run_label = (parent.parent.name if parent.name == "data" else parent.name) or "prediction"
    # target_grib / target_label are the lane-config-level way to specify the high-res
    # truth target (e.g. IEKM for o1280_o2560). They are resolved into extra_grib_references
    # internally so callers never need to use the lower-level dict directly.
    target_grib = eval_config.get("target_grib")
    target_label = eval_config.get("target_label", "TARGET")
    if extra_grib_references is None and target_grib:
        extra_grib_references = {target_label: target_grib}

    event_names = eval_config.get("events", [])
    if not event_names:
        LOG.warning("No TC events configured in lane_config['tc']['events']")
        return output_dir

    pred_files = _pred_files_as_tuples(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    payload = {"events": {}}

    for event_name in event_names:
        event = _event_from_config(event_name)
        event_pred_files = select_prediction_files_for_event(pred_files, event)
        if not event_pred_files:
            LOG.info("Skipping event=%s: no matching prediction files", event_name)
            continue

        exp_cfg = None
        if analysis_expid:
            exp_cfg = TCExperimentConfig(
                analysis_expid=analysis_expid,
                base_tc_dir=grib_dir,
                reference_expids=reference_expids,
            )

        curves = load_curves_for_event(
            event,
            prediction_dir=predictions_dir,
            grib_dir=grib_dir,
            experiment_config=exp_cfg,
            extra_grib_references=extra_grib_references,
            support_mode=support_mode,
            run_label=run_label,
            pred_files=event_pred_files,
        )

        if not curves:
            LOG.warning("No curves loaded for event=%s, skipping", event_name)
            continue

        # Load input (x_interp) and target (y) curves directly from NetCDF files
        input_label = eval_config.get("input_label")
        if input_label and event_pred_files:
            try:
                curves[input_label] = load_prediction_curves(
                    event_pred_files,
                    bbox=event.bbox,
                    support_mode="native",
                    prediction_var="x_interp",
                )
            except Exception:
                LOG.warning("Failed to load x_interp curve for event=%s", event_name, exc_info=True)

        target_nc_label = eval_config.get("target_nc_label")
        if target_nc_label and event_pred_files:
            try:
                curves[target_nc_label] = load_prediction_curves(
                    event_pred_files,
                    bbox=event.bbox,
                    support_mode="native",
                    prediction_var="y",
                )
            except Exception:
                LOG.warning("Failed to load y curve for event=%s", event_name, exc_info=True)

        # Determine analysis key
        oper_key = analysis_expid
        if oper_key and oper_key not in curves:
            LOG.warning("Analysis key %s not in curves for event=%s", oper_key, event_name)
            # Fall back: if only prediction curves, skip stats that need analysis
            oper_key = None

        # Rename analysis key if a display label is configured
        analysis_display_label = eval_config.get("analysis_display_label")
        if analysis_display_label and oper_key and oper_key in curves:
            curves[analysis_display_label] = curves.pop(oper_key)
            oper_key = analysis_display_label

        if oper_key:
            plot_cfg = PLOT_CONFIGS.get(event_name, TCPlotConfig())
            event_stats = compute_event_stats(
                curves,
                analysis_key=oper_key,
                plot_config=plot_cfg,
            )
        else:
            # Prediction-only mode: store raw curve metadata without analysis-anchored stats
            days, steps = event_days_steps(event_pred_files)
            event_stats = {
                "event": event_name,
                "selected_days": days,
                "steps_hours": steps,
                "prediction_only": True,
            }

        event_stats["event"] = event_name
        payload["events"][event_name] = event_stats

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)

    LOG.info("TC stats written to %s", stats_path)

    # Member maps (optional, controlled by eval_config["member_maps"])
    mm_cfg = eval_config.get("member_maps") or {}
    if mm_cfg.get("enabled"):
        mm_dates = mm_cfg.get("dates") or []
        mm_steps = mm_cfg.get("steps") or [24, 120]
        mm_members = mm_cfg.get("members") or list(range(10))
        mm_events = mm_cfg.get("events") or list(event_names)
        mm_outdir = output_dir / "member_maps"
        for date in mm_dates:
            try:
                run_member_maps(
                    predictions_dir=str(predictions_dir),
                    outdir=str(mm_outdir),
                    run_label=run_label,
                    event_names=mm_events,
                    date=date,
                    steps=mm_steps,
                    members=mm_members,
                )
                LOG.info("Member maps written for date=%s to %s", date, mm_outdir)
            except Exception:
                LOG.error("Member maps failed for date=%s", date, exc_info=True)

    return output_dir
