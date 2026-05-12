"""Composable building blocks + CLI for TC evaluation workflows."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from eval.paths import reference_tc_dir

from .data_types import BoundingBox, CurveVectors, SupportMode
from .events import EVENTS, TCEvent
from .experiment_config import EXPERIMENT_CONFIGS, TCExperimentConfig
from .loading_grib import load_grib_curve_from_paths, load_grib_curves, regridded_target_points
from .loading_predictions import (
    analysis_dates_for_event,
    discover_prediction_files,
    event_days_steps,
    forecast_dates_for_event,
    load_prediction_curves,
    load_prediction_member_fields,
    select_prediction_files_for_event,
)
from .member_plot import _plot_member_page
from .pdf_plot import plot_pdf_ratios
from .plot_config import PLOT_CONFIGS, TCPlotConfig
from .stats import _finite_1d, extreme_tail_table, summary_stats, tail_summary, variable_stats

LOG = logging.getLogger(__name__)

MAX_DISPLAY_LABEL_CHARS = 24


def cut_display_label(label: str | None, *, fallback: str) -> str:
    """Return compact legend text while keeping the long run label as the key."""
    if label and label.strip():
        text = label.strip()
    else:
        tokens = [t for t in re.split(r"[_\s-]+", fallback) if t]
        short_hash = ""
        stack = ""
        sampler = ""
        extra = ""
        for token in tokens:
            low = token.lower()
            if low == "manual":
                continue
            if not short_hash and re.fullmatch(r"[0-9a-f]{8,}", low):
                short_hash = low[:4]
                continue
            if low in {"old", "new"}:
                stack = low
                continue
            if not sampler and low.startswith("piecewise"):
                sampler = "pw" + low.removeprefix("piecewise")
                continue
            if not sampler and re.fullmatch(r"pw\d+", low):
                sampler = low
                continue
            if not sampler and low.startswith("karras") and low != "karras":
                sampler = "k" + low.removeprefix("karras")
                continue
            if not sampler and low == "karras":
                sampler = "k"
                continue
            if sampler == "k" and re.fullmatch(r"n\d+", low):
                sampler = "k" + low[1:]
                continue
            if not sampler and low.startswith("lognorm"):
                sampler = low
                continue
            if low.startswith("sigmax"):
                extra = low.replace("sigmax100k", "s100k")
                continue
            if low.startswith("oldlike"):
                extra = low.replace("oldlike200k", "old200k").replace("oldlike100kft", "old100k")
                continue
        parts = [p for p in [short_hash, stack, sampler, extra] if p]
        text = " ".join(parts) if parts else fallback
    text = text.replace("piecewise", "pw")
    text = text.replace("karras", "k")
    text = text.replace("sigmax100k", "s100k")
    text = text.replace("oldlike200k", "old200k")
    text = text.replace("oldlike100kft", "old100k")
    text = re.sub(r"\b20\d{6}\b", "", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = text.split()
    if len(parts) > 4:
        text = " ".join(parts[:4])
    if len(text) > MAX_DISPLAY_LABEL_CHARS:
        text = text[:MAX_DISPLAY_LABEL_CHARS].rstrip()
    return text or fallback[:MAX_DISPLAY_LABEL_CHARS]


# --- Building blocks ---



def load_curves_for_event(
    event: TCEvent,
    *,
    prediction_dir: Path | None = None,
    grib_dir: str | None = None,
    experiment_config: TCExperimentConfig | None = None,
    extra_grib_references: dict[str, str] | None = None,
    support_mode: SupportMode = "native",
    prediction_var: str = "y_pred",
    run_label: str = "",
    ml_expids: list[str] | None = None,
    forecast_dates: list[str] | None = None,
    analysis_dates: list[str] | None = None,
    steps: list[int] | None = None,
    max_pf_members: int | None = None,
    regrid_resolution: float = 0.25,
    pred_files: list | None = None,
) -> dict[str, CurveVectors]:
    """Load curves from any combination of sources for one event.

    Predictions, standard GRIB references, and custom GRIB paths can all be mixed.
    """
    curves: dict[str, CurveVectors] = {}
    exp_cfg = experiment_config
    analysis_expid = exp_cfg.analysis_expid if exp_cfg else None
    base_tc_dir = grib_dir or (exp_cfg.base_tc_dir if exp_cfg else None)

    # Discover prediction files for this event
    event_pred_files = pred_files
    days_from_preds = None
    steps_from_preds = None
    if prediction_dir and event_pred_files is None:
        all_pred_files = discover_prediction_files(prediction_dir)
        event_pred_files = select_prediction_files_for_event(all_pred_files, event)

    if event_pred_files:
        days_from_preds, steps_from_preds = event_days_steps(event_pred_files)

    # Compute effective dates
    eff_forecast_dates = forecast_dates
    eff_analysis_dates = analysis_dates
    eff_steps = steps
    if eff_forecast_dates is None and days_from_preds:
        eff_forecast_dates = forecast_dates_for_event(event, days_from_preds)
    if eff_analysis_dates is None and days_from_preds:
        eff_analysis_dates = analysis_dates_for_event(event, days_from_preds)
    if eff_steps is None and steps_from_preds:
        eff_steps = steps_from_preds

    # Load GRIB reference curves (analysis + configured references)
    if base_tc_dir and analysis_expid:
        ref_expids = list(exp_cfg.reference_expids) if exp_cfg else []
        if ml_expids:
            ref_expids = list(dict.fromkeys([*ml_expids, *ref_expids]))
        grib_curves = load_grib_curves(
            dir_data_base=base_tc_dir,
            event_name=event.name,
            analysis_expid=analysis_expid,
            analysis_dates=eff_analysis_dates or list(event.analysis_dates),
            forecast_dates=eff_forecast_dates or forecast_dates_for_event(event),
            reference_expids=ref_expids,
            support_mode=support_mode,
            bbox=event.bbox if support_mode == "regridded" else None,
            regrid_resolution=regrid_resolution,
            steps=eff_steps,
            max_pf_members=max_pf_members,
        )
        curves.update(grib_curves)

    # Load prediction curve
    if event_pred_files and run_label:
        target_lon = None
        target_lat = None
        if support_mode == "regridded" and base_tc_dir and analysis_expid:
            ad = eff_analysis_dates or list(event.analysis_dates)
            sample_path = f"{base_tc_dir}/{event.name}/surface_an_{analysis_expid}_{ad[0]}.grib"
            target_lon, target_lat = regridded_target_points(
                event.bbox, regrid_resolution, sample_path,
            )
        pred_curve = load_prediction_curves(
            event_pred_files,
            bbox=event.bbox,
            support_mode=support_mode,
            target_lon=target_lon,
            target_lat=target_lat,
            prediction_var=prediction_var,
        )
        curves[run_label] = pred_curve

    # Load extra GRIB references
    if extra_grib_references:
        for label, glob_path in extra_grib_references.items():
            curves[label] = load_grib_curve_from_paths(
                glob_path,
                support_mode=support_mode,
                bbox=event.bbox if support_mode == "regridded" else None,
                regrid_resolution=regrid_resolution,
            )

    return curves


def compute_event_stats(
    curves: dict[str, CurveVectors],
    *,
    analysis_key: str,
    plot_config: TCPlotConfig,
    curve_order: list[str] | None = None,
    extreme_mslp_range: tuple[float, float] = (980.0, 990.0),
    extreme_wind_threshold: float = 25.0,
) -> dict:
    """Compute all statistics for one event's curves. No matplotlib."""
    oper_curve = curves[analysis_key]
    oper_msl = _finite_1d(oper_curve.msl)
    oper_wind = _finite_1d(oper_curve.wind)

    if curve_order is None:
        curve_order = [k for k in curves if k != analysis_key]
    else:
        curve_order = [k for k in curve_order if k in curves and k != analysis_key]

    xbins_msl = np.arange(*plot_config.mslp_bin_range)
    mids_msl = (xbins_msl[:-1] + xbins_msl[1:]) / 2.0
    xbins_wind = np.arange(*plot_config.wind_bin_range)
    mids_wind = (xbins_wind[:-1] + xbins_wind[1:]) / 2.0

    hist_oper_msl, _ = np.histogram(oper_msl, bins=xbins_msl, density=True)
    hist_oper_wind, _ = np.histogram(oper_wind, bins=xbins_wind, density=True)

    s_oper_msl = tail_summary(oper_msl, tail="low")
    s_oper_wind = tail_summary(oper_wind, tail="high")

    LOG.info(
        "MSLP OPER support [%.2f, %.2f] hPa | tail_index=%.3g extreme_index=%.3g",
        float(oper_msl.min()) if oper_msl.size else float("nan"),
        float(oper_msl.max()) if oper_msl.size else float("nan"),
        s_oper_msl.get("tail_index", float("nan")),
        s_oper_msl.get("extreme_index", float("nan")),
    )
    LOG.info(
        "WIND OPER support [%.2f, %.2f] m/s | tail_index=%.3g extreme_index=%.3g",
        float(oper_wind.min()) if oper_wind.size else float("nan"),
        float(oper_wind.max()) if oper_wind.size else float("nan"),
        s_oper_wind.get("tail_index", float("nan")),
        s_oper_wind.get("extreme_index", float("nan")),
    )

    event_stats: dict = {
        "analysis_key": analysis_key,
        "curve_order": list(curve_order),
        "variables": {
            "mslp_hpa": {
                "bin_edges": xbins_msl.tolist(),
                "bin_mids": mids_msl.tolist(),
                "oper": {"summary": summary_stats(oper_msl), "tail": s_oper_msl},
                "oper_histogram": hist_oper_msl.tolist(),
                "curves": {},
            },
            "wind10m_ms": {
                "bin_edges": xbins_wind.tolist(),
                "bin_mids": mids_wind.tolist(),
                "oper": {"summary": summary_stats(oper_wind), "tail": s_oper_wind},
                "oper_histogram": hist_oper_wind.tolist(),
                "curves": {},
            },
        },
    }

    extreme_series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        analysis_key: (oper_msl, oper_wind),
    }

    all_msl_vals = [oper_msl]
    all_wind_vals = [oper_wind]

    for curve_key in curve_order:
        curve = curves[curve_key]
        vals_msl = _finite_1d(curve.msl)
        hist_msl, msl_stats = variable_stats(
            vals_msl,
            hist_ref=hist_oper_msl,
            bins=xbins_msl,
            bin_width=plot_config.mslp_bin_range[2],
            tail="low",
        )
        LOG.info(
            "MSLP %-24s support [%.2f, %.2f] | tail=%.3g ext=%.3g",
            curve_key,
            float(vals_msl.min()) if vals_msl.size else float("nan"),
            float(vals_msl.max()) if vals_msl.size else float("nan"),
            msl_stats["tail"].get("tail_index", float("nan")),
            msl_stats["tail"].get("extreme_index", float("nan")),
        )
        event_stats["variables"]["mslp_hpa"]["curves"][curve_key] = {
            "histogram": hist_msl.tolist(),
            **msl_stats,
        }

        vals_wind = _finite_1d(curve.wind)
        hist_wind, wind_stats = variable_stats(
            vals_wind,
            hist_ref=hist_oper_wind,
            bins=xbins_wind,
            bin_width=plot_config.wind_bin_range[2],
            tail="high",
        )
        LOG.info(
            "WIND %-24s support [%.2f, %.2f] | tail=%.3g ext=%.3g",
            curve_key,
            float(vals_wind.min()) if vals_wind.size else float("nan"),
            float(vals_wind.max()) if vals_wind.size else float("nan"),
            wind_stats["tail"].get("tail_index", float("nan")),
            wind_stats["tail"].get("extreme_index", float("nan")),
        )
        event_stats["variables"]["wind10m_ms"]["curves"][curve_key] = {
            "histogram": hist_wind.tolist(),
            **wind_stats,
        }
        extreme_series[curve_key] = (vals_msl, vals_wind)
        all_msl_vals.append(vals_msl)
        all_wind_vals.append(vals_wind)

    # Data ranges for auto-crop in rendering
    all_msl = np.concatenate(all_msl_vals) if all_msl_vals else np.array([])
    all_wind = np.concatenate(all_wind_vals) if all_wind_vals else np.array([])
    if all_msl.size:
        event_stats["variables"]["mslp_hpa"]["data_range_msl"] = [float(all_msl.min()), float(all_msl.max())]
    if all_wind.size:
        event_stats["variables"]["wind10m_ms"]["data_range_wind"] = [float(all_wind.min()), float(all_wind.max())]

    event_stats["extreme_tail"] = extreme_tail_table(
        extreme_series,
        mslp_range=extreme_mslp_range,
        wind_threshold=extreme_wind_threshold,
    )
    return event_stats


def render_event_figure(
    event_stats: dict,
    *,
    plot_config: TCPlotConfig,
    exp_labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Render one event's stats as a PDF ratio figure."""
    return plot_pdf_ratios(plot_config, event_stats=event_stats, exp_labels=exp_labels)


# --- Convenience workflows ---


def run_tc_pdf(
    *,
    outdir: str,
    event_names: list[str] | None = None,
    prediction_dir: str | None = None,
    grib_dir: str = str(reference_tc_dir("o96_o320")),
    run_label: str = "",
    display_label: str | None = None,
    out_name: str = "",
    support_mode: SupportMode = "regridded",
    extra_grib_references: dict[str, str] | None = None,
    extra_reference_expids: list[str] | None = None,
    prediction_var: str = "y_pred",
    analysis_expid: str | None = None,
    plot_title_override: str | None = None,
    ml_expids: list[str] | None = None,
    max_pf_members: int | None = None,
    extreme_mslp_range: tuple[float, float] = (980.0, 990.0),
    extreme_wind_threshold: float = 25.0,
) -> str:
    """Main workflow: load from any sources -> stats -> render -> write PDF + JSON.

    Works for predictions, GRIBs, or a mix of both.
    """
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(prediction_dir).expanduser().resolve() if prediction_dir else None
    pred_files_all = None
    if pred_dir:
        pred_files_all = discover_prediction_files(pred_dir)
        if not pred_files_all:
            raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    selected_events = event_names or list(EVENTS.keys())

    if not out_name:
        label_part = run_label or "grib"
        out_name = f"tc_normed_pdfs_{label_part}_{support_mode}_from_predictions.pdf"
    out_pdf = out_dir / out_name
    out_stats = out_dir / f"{out_pdf.stem}.stats.json"
    plot_label = cut_display_label(display_label, fallback=run_label or "ML")

    payload: dict[str, object] = {
        "run_label": run_label,
        "display_label": plot_label,
        "raw_display_label": display_label or "",
        "predictions_dir": str(pred_dir) if pred_dir else "",
        "pdf_file": str(out_pdf),
        "support_mode": support_mode,
        "prediction_var": prediction_var,
        "events": {},
    }

    rendered_events = 0
    with PdfPages(out_pdf) as pdf:
        for event_name in selected_events:
            if event_name not in EVENTS:
                LOG.warning("Unknown event=%s, skipping", event_name)
                continue
            event = EVENTS[event_name]

            # Get experiment config, with optional analysis_expid override
            exp_cfg = EXPERIMENT_CONFIGS.get(event_name)
            if exp_cfg is None:
                LOG.warning(
                    "No experiment config for event=%s — skipping GRIB references", event_name
                )
            if analysis_expid:
                if exp_cfg is None:
                    exp_cfg = TCExperimentConfig(analysis_expid=analysis_expid, base_tc_dir=grib_dir)
                else:
                    exp_cfg = replace(exp_cfg, analysis_expid=analysis_expid)

            # Add extra_reference_expids to exp_cfg
            if exp_cfg and extra_reference_expids:
                merged_refs = list(dict.fromkeys([*exp_cfg.reference_expids, *extra_reference_expids]))
                exp_cfg = replace(exp_cfg, reference_expids=tuple(merged_refs))

            # Validate: analysis_expid is required to anchor TC PDF statistics.
            if exp_cfg and exp_cfg.analysis_expid is None:
                raise ValueError(
                    f"No analysis_expid configured for event={event_name}. "
                    "Pass --analysis-expid explicitly (e.g. OPER_O320_0001)."
                )

            # Get plot config, with optional title override
            plot_cfg = PLOT_CONFIGS.get(event_name, TCPlotConfig())
            if plot_title_override:
                plot_cfg = replace(plot_cfg, plot_title=plot_title_override)

            # Filter prediction files for this event
            event_pred_files = None
            if pred_files_all:
                event_pred_files = select_prediction_files_for_event(pred_files_all, event)
                if not event_pred_files:
                    LOG.info("Skipping event=%s: no matching prediction files", event_name)
                    continue

            curves = load_curves_for_event(
                event,
                prediction_dir=pred_dir,
                grib_dir=grib_dir,
                experiment_config=exp_cfg,
                extra_grib_references=extra_grib_references,
                support_mode=support_mode,
                prediction_var=prediction_var,
                run_label=run_label,
                ml_expids=ml_expids,
                max_pf_members=max_pf_members or (10 if pred_dir else None),
                regrid_resolution=plot_cfg.regrid_resolution,
                pred_files=event_pred_files,
            )

            if not curves:
                LOG.warning("No curves loaded for event=%s, skipping", event_name)
                continue

            # Determine analysis key and curve order
            oper_key = exp_cfg.analysis_expid if exp_cfg else None
            if oper_key not in curves:
                LOG.warning("Analysis key %s not in loaded curves for event=%s, skipping", oper_key, event_name)
                continue

            curve_order = [k for k in curves if k != oper_key]

            event_stats = compute_event_stats(
                curves,
                analysis_key=oper_key,
                plot_config=plot_cfg,
                curve_order=curve_order,
                extreme_mslp_range=extreme_mslp_range,
                extreme_wind_threshold=extreme_wind_threshold,
            )
            event_stats["event"] = event_name
            event_stats["year"] = event.year
            event_stats["month"] = event.month
            event_stats["dates"] = list(event.dates)
            event_stats["analysis_dates"] = list(event.analysis_dates)

            if event_pred_files:
                days, steps_h = event_days_steps(event_pred_files)
                event_stats["selected_days"] = days
                event_stats["steps_hours"] = steps_h

            exp_labels = {}
            if run_label:
                exp_labels[run_label] = plot_label
            if extra_reference_expids:
                for expid in extra_reference_expids:
                    exp_labels[expid] = expid.replace("ENFO_O320_", "")

            fig = render_event_figure(event_stats, plot_config=plot_cfg, exp_labels=exp_labels)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            payload["events"][event_name] = event_stats
            rendered_events += 1

    if rendered_events == 0:
        raise RuntimeError("No matching events found.")

    with open(out_stats, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)

    LOG.info("Saved TC PDF: %s", out_pdf)
    LOG.info("Saved TC stats JSON: %s", out_stats)
    return str(out_pdf)


def run_member_maps(
    *,
    predictions_dir: str,
    outdir: str,
    run_label: str,
    display_label: str | None = None,
    event_names: list[str] | None = None,
    date: str,
    steps: list[int] | None = None,
    members: list[int] | None = None,
) -> list[str]:
    """Member spatial maps workflow."""
    display_label = display_label or run_label
    steps = steps or [24, 120]
    members = members or [0, 1, 2, 3, 4]

    pred_dir = Path(predictions_dir).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = discover_prediction_files(pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")

    selected_events = event_names or list(EVENTS.keys())
    generated: list[str] = []

    for event_name in selected_events:
        if event_name not in EVENTS:
            LOG.warning("Unknown event=%s, skipping", event_name)
            continue
        event = EVENTS[event_name]
        exp_cfg = EXPERIMENT_CONFIGS.get(event_name)
        plot_cfg = PLOT_CONFIGS.get(event_name, TCPlotConfig())

        event_pred_files = select_prediction_files_for_event(pred_files, event)
        if not event_pred_files:
            LOG.info("Skipping event=%s: no matching prediction files", event_name)
            continue

        date_int = int(date)
        step_files = [
            (p, ymd, step) for p, ymd, step in event_pred_files
            if ymd == date_int and step in steps
        ]
        if not step_files:
            LOG.warning(
                "Skipping event=%s: no prediction files for date=%s steps=%s",
                event_name, date, steps,
            )
            continue

        safe_label = display_label.replace(" ", "_").replace("/", "_")
        pdf_name = f"tc_members_{event_name}_{safe_label}_{date}.pdf"
        pdf_path = out_dir / pdf_name

        with PdfPages(pdf_path) as pdf:
            for nc_path, ymd, step in sorted(step_files, key=lambda x: x[2]):
                LOG.info("Loading event=%s date=%s step=%d", event_name, date, step)
                try:
                    fields = load_prediction_member_fields(
                        nc_path, event.bbox, members,
                        regrid_resolution=plot_cfg.regrid_resolution,
                    )
                except Exception as exc:
                    LOG.warning("Failed to load %s: %s", nc_path.name, exc)
                    continue
                for mi, mbr in enumerate(members):
                    LOG.info("  Plotting member %d (step=%dh)", mbr, step)
                    fig = _plot_member_page(
                        fields,
                        bbox=event.bbox,
                        plot_config=plot_cfg,
                        exp_config=exp_cfg,
                        member_idx=mi,
                        member_label=mbr,
                        step_hours=step,
                        date_str=date,
                        display_label=display_label,
                        event_name=event_name,
                    )
                    pdf.savefig(fig, dpi=200, bbox_inches="tight")
                    plt.close(fig)

        LOG.info("Saved TC member maps PDF: %s", pdf_path)
        generated.append(str(pdf_path))

    return generated


# --- Legacy GRIB-only member plots ---

def run_tc_member_plots_legacy(
    *,
    expver: str,
    outdir: str,
    base_dir: str = str(reference_tc_dir("o96_o320")),
    members: list[int] | None = None,
    year: str = "2023",
    month: str = "08",
    resol: float = 0.25,
) -> list[str]:
    """Legacy GRIB-based member plots using DataRetriever (preserved for backward compat)."""
    from .tools.loading_data import DataRetriever

    members = members or [1, 2, 5, 7, 9]
    out_root = Path(outdir).expanduser().resolve() / "tc_members"
    out_root.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    CASES = [
        {"name": "dora", "lat": (5.5, 23.0), "lon": (-158.75, -141.0), "dates": [6], "time": 1,
         "msl_levels": np.linspace(985, 1015, 31), "wind_levels": np.linspace(0, 30, 31)},
        {"name": "fernanda", "lat": (1.0, 29.0), "lon": (-134.0, -105.0), "dates": [13], "time": 2,
         "msl_levels": np.linspace(985, 1015, 31), "wind_levels": np.linspace(0, 30, 31)},
        {"name": "hilary", "lat": (11.0, 29.0), "lon": (-120.0, -100.0), "dates": [17], "time": 1,
         "msl_levels": np.linspace(965, 1015, 31), "wind_levels": np.linspace(0, 35, 31)},
        {"name": "idalia", "lat": (11.0, 39.01), "lon": (-99.0, -71.01), "dates": [28], "time": 1,
         "msl_levels": np.linspace(985, 1015, 31), "wind_levels": np.linspace(0, 30, 31)},
        {"name": "franklin", "lat": (11.0, 39.0), "lon": (-79.0, -51.0), "dates": [28], "time": 1,
         "msl_levels": np.linspace(985, 1015, 31), "wind_levels": np.linspace(0, 30, 31)},
    ]

    analysis = "OPER_O320_0001"
    expid_enfo_o320 = "ENFO_O320_0001"
    expid_eefo_o96 = "EEFO_O96_0001"

    def _candidate_expids(ev: str) -> list[str]:
        out = [ev]
        if not ev.startswith("ENFO_O320_"):
            out.insert(0, f"ENFO_O320_{ev}")
        return list(dict.fromkeys(out))

    def _resolve_expid_for_case(*, case_name, day, ev):
        for expid in _candidate_expids(ev):
            p = Path(base_dir) / case_name / f"surface_pf_{expid}_{year}{month}{int(day):02d}.grib"
            if p.exists():
                return expid
        return ev

    for case in CASES:
        lat = np.arange(case["lat"][0], case["lat"][1] + resol / 4, resol)
        lon = np.arange(case["lon"][0], case["lon"][1] + resol / 4, resol)
        retriever = DataRetriever(
            f"{base_dir}/{case['name']}", case["dates"], year, month, resol,
            float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()),
        )
        ml_expid = _resolve_expid_for_case(case_name=case["name"], day=case["dates"][0], ev=expver)
        try:
            msl, wind10m = retriever.retrieve_all_data(analysis, expid_enfo_o320, expid_eefo_o96, [ml_expid])
        except Exception as exc:
            LOG.warning("Skipping case %s: %s", case["name"], exc)
            continue

        lon2, lat2 = np.meshgrid(lon, lat)
        pf_keys = [k for k in msl.keys() if "OPER" not in k]
        if not pf_keys:
            continue
        max_member = min(int(msl[k].shape[1]) - 1 for k in pf_keys)
        safe_members = [m for m in members if 0 <= m <= max_member]
        if not safe_members:
            continue

        t_idx = int(case["time"])
        max_time = min(int(msl[k].shape[2]) for k in pf_keys) - 1
        if t_idx > max_time:
            continue

        msl_data = {k: v for k, v in msl.items() if "OPER" not in k}
        wind_data = {k: v for k, v in wind10m.items() if "OPER" not in k}
        if not msl_data or not wind_data:
            continue

        from cartopy import crs as crs_mod
        for var_key, data_dict, levels, title_prefix in [
            ("msl", msl_data, case["msl_levels"], "Mean Sea Level Pressure"),
            ("wind10m", wind_data, case["wind_levels"], f"Wind Speed 10m - {month} Lead Time {t_idx}"),
        ]:
            filename = f"{case['name']}_{var_key}_fields_{case['dates'][0]}_{month}_step{t_idx}.png"
            datasets = [(data, key) for key, data in data_dict.items()]
            ncols, nrows = len(datasets), len(safe_members)
            fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows),
                                    subplot_kw={"projection": crs_mod.PlateCarree()})
            fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=-0.1, hspace=0.15)
            axs_arr = np.array(axs).reshape(nrows, ncols)
            images = []
            for row, member in enumerate(safe_members):
                for col, (data, label) in enumerate(datasets):
                    ax = axs_arr[row, col]
                    arr = data[0, member, t_idx][10:100, 10:100]
                    im = ax.pcolormesh(lon2[10:100, 10:100], lat2[10:100, 10:100], arr,
                                       transform=crs_mod.PlateCarree(),
                                       vmin=float(levels[0]), vmax=float(levels[-1]),
                                       shading="gouraud", cmap="viridis")
                    ax.contour(lon2[10:100, 10:100], lat2[10:100, 10:100], arr,
                               transform=crs_mod.PlateCarree(), levels=levels,
                               colors="black", linewidths=0.5)
                    ax.set_title(f"mbr{member} {label}", fontsize=12)
                    ax.coastlines()
                    ax.grid(color="white", linestyle="--", linewidth=0.5)
                    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
                    gl.top_labels = False
                    gl.right_labels = False
                    images.append(im)
            for row in range(nrows):
                fig.colorbar(images[row * ncols], ax=axs_arr[row], orientation="vertical", shrink=0.8, pad=0.02)
            fig.suptitle(title_prefix, fontsize=14)
            out_path = out_root / filename
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(out_path))
        LOG.info("Generated TC member plots for %s", case["name"])

    return generated


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --- CLI ---


def _parse_members(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_extra_grib_refs(raw: str) -> dict[str, str]:
    """Parse 'LABEL1=path1,LABEL2=path2' into a dict."""
    refs: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
            refs[label.strip()] = path.strip()
        else:
            refs[item] = item
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TC evaluation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- pdf subcommand ---
    pdf_parser = subparsers.add_parser("pdf", help="Generate TC normalized PDF plots.")
    pdf_parser.add_argument("--predictions-dir", default=None)
    pdf_parser.add_argument("--outdir", required=True)
    pdf_parser.add_argument("--run-label", default="")
    pdf_parser.add_argument("--display-label", default="")
    pdf_parser.add_argument("--prediction-var", default="y_pred")
    pdf_parser.add_argument("--plot-title", default="")
    pdf_parser.add_argument("--out-name", default="")
    pdf_parser.add_argument("--base-tc-dir", default=str(reference_tc_dir("o96_o320")))
    pdf_parser.add_argument("--extra-reference-expids", default="",
                            help="Comma-separated extra reference expids")
    pdf_parser.add_argument("--extra-grib-reference", default="",
                            help="label=path pairs, comma-separated")
    pdf_parser.add_argument("--support-mode", choices=["native", "regridded"], default="regridded")
    pdf_parser.add_argument("--events", default="")
    pdf_parser.add_argument("--analysis-expid", default="")
    pdf_parser.add_argument("--iekm-target-grib-path", default="",
                            help="Path to IEKM target GRIBs; 'none' disables")
    pdf_parser.add_argument("--iekm-label", default="IEKM_TARGET")
    pdf_parser.add_argument("--ml-expids", default="",
                            help="Comma-separated ML GRIB expids (GRIB-only mode)")
    pdf_parser.add_argument("--log-level", default="INFO")

    # --- member-maps subcommand ---
    mm_parser = subparsers.add_parser("member-maps", help="Generate TC member spatial maps.")
    mm_parser.add_argument("--predictions-dir", required=True)
    mm_parser.add_argument("--outdir", required=True)
    mm_parser.add_argument("--run-label", required=True)
    mm_parser.add_argument("--display-label", default="")
    mm_parser.add_argument("--events", default="")
    mm_parser.add_argument("--date", required=True)
    mm_parser.add_argument("--steps", default="24,120")
    mm_parser.add_argument("--members", default="0,1,2,3,4")
    mm_parser.add_argument("--log-level", default="INFO")

    # --- legacy-members subcommand (GRIB-based) ---
    leg_parser = subparsers.add_parser("legacy-members", help="Legacy GRIB-based member plots.")
    leg_parser.add_argument("--expver", required=True)
    leg_parser.add_argument("--outdir", required=True)
    leg_parser.add_argument("--base-dir", default=str(reference_tc_dir("o96_o320")))
    leg_parser.add_argument("--members", default="1,2,5,7,9")
    leg_parser.add_argument("--year", default="2023")
    leg_parser.add_argument("--month", default="08")
    leg_parser.add_argument("--resol", type=float, default=0.25)
    leg_parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.command == "pdf":
        event_names = [v.strip() for v in args.events.split(",") if v.strip()] or None
        extra_reference_expids = [v.strip() for v in args.extra_reference_expids.split(",") if v.strip()]
        ml_expids = [v.strip() for v in args.ml_expids.split(",") if v.strip()] or None

        # Handle IEKM as an extra GRIB reference
        extra_grib_refs = _parse_extra_grib_refs(args.extra_grib_reference) if args.extra_grib_reference else {}
        if args.iekm_target_grib_path and args.iekm_target_grib_path.lower() != "none":
            extra_grib_refs[args.iekm_label] = args.iekm_target_grib_path

        run_tc_pdf(
            outdir=args.outdir,
            event_names=event_names,
            prediction_dir=args.predictions_dir,
            grib_dir=args.base_tc_dir,
            run_label=args.run_label,
            display_label=args.display_label or None,
            out_name=args.out_name,
            support_mode=args.support_mode,
            extra_grib_references=extra_grib_refs or None,
            extra_reference_expids=extra_reference_expids or None,
            prediction_var=args.prediction_var,
            analysis_expid=args.analysis_expid or None,
            plot_title_override=args.plot_title or None,
            ml_expids=ml_expids,
        )

    elif args.command == "member-maps":
        event_names = [v.strip() for v in args.events.split(",") if v.strip()] or None
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
        run_member_maps(
            predictions_dir=args.predictions_dir,
            outdir=args.outdir,
            run_label=args.run_label,
            display_label=args.display_label or None,
            event_names=event_names,
            date=args.date,
            steps=steps,
            members=_parse_members(args.members),
        )

    elif args.command == "legacy-members":
        run_tc_member_plots_legacy(
            expver=args.expver,
            outdir=args.outdir,
            base_dir=args.base_dir,
            members=_parse_members(args.members),
            year=args.year,
            month=args.month,
            resol=args.resol,
        )


if __name__ == "__main__":
    main()
