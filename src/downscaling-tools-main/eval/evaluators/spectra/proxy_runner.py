#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from eval._backends.spectra.calibrate_fast_spectra_proxy import (
        cl_from_unstructured as _healpy_cl_from_unstructured,
    )
except ImportError:
    _healpy_cl_from_unstructured = None

from eval.checkpoint_interpolation import (
    CheckpointResidualInterpolator,
    resolve_checkpoint_path,
)


SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT = 100.0


@dataclass(frozen=True)
class ScopeSpec:
    name: str
    title: str
    curve_a_label: str
    curve_b_label: str
    pdf_name: str
    curve_c_label: str | None = None  # input curve; None means not plotted for this scope


SCOPE_SPECS = {
    "full_field": ScopeSpec(
        name="full_field",
        title="full-field spectra",
        curve_a_label="prediction mean",
        curve_b_label="truth mean",
        curve_c_label="input (interp) mean",
        pdf_name="spectra_{state}.pdf",
    ),
    "residual": ScopeSpec(
        name="residual",
        title="residual spectra",
        curve_a_label="prediction residual mean",
        curve_b_label="truth residual mean",
        curve_c_label=None,  # input residual is zero by definition
        pdf_name="spectra_residual_{state}.pdf",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute prediction-vs-truth spectra diagnostics from predictions_*.nc files "
            "for both reconstructed full fields and canonical residuals."
        )
    )
    p.add_argument("--predictions-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--weather-states", default="10u,10v,2t,msl,t_850,z_500,tp")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=319)
    p.add_argument(
        "--prediction-var",
        default="y_pred",
        help="Dataset variable to compare against truth for full-field spectra (default: y_pred).",
    )
    p.add_argument(
        "--checkpoint-path",
        default="",
        help=(
            "Optional base checkpoint path used to reconstruct the interpolated high-resolution "
            "input for residual spectra. When omitted, the helper resolves it from predictions "
            "metadata or the run EXPERIMENT_CONFIG.yaml."
        ),
    )
    p.add_argument(
        "--spectra-method",
        choices=["auto", "healpy", "fft_proxy"],
        default="auto",
        help=(
            "How to convert unstructured lat/lon fields into spectra. "
            "'auto' prefers healpy when available and otherwise falls back to fft_proxy."
        ),
    )
    p.add_argument(
        "--member-aggregation",
        choices=["per-file-mean", "raw-members"],
        default="per-file-mean",
        help=(
            "How to treat ensemble members. "
            "'per-file-mean' averages member spectra within each prediction file first, "
            "so one curve contributes per case/file. "
            "'raw-members' keeps one curve per member."
        ),
    )
    p.add_argument(
        "--show-individual-curves",
        action="store_true",
        help="Overlay every aggregated case curve in the background instead of plotting only mean curves.",
    )
    p.add_argument(
        "--score-wavenumber-min-exclusive",
        type=float,
        default=SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT,
        help=(
            "Only wavenumbers strictly above this threshold contribute to relative_l2_mean_curve. "
            f"Default: {SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT:g}."
        ),
    )
    p.add_argument(
        "--consolidated-pdf",
        default="",
        help=(
            "Path to a single consolidated multi-page PDF merging all per-variable and per-scope "
            "plots into one file. When set, the consolidated PDF is written in addition to the "
            "per-variable PDFs. Typically set to <RUN_ROOT>/spectra_proxy.pdf or spectra_ecmwf.pdf."
        ),
    )
    p.add_argument(
        "--plots-dir",
        default="",
        help=(
            "When set, per-variable PDFs are also copied/linked to this directory with "
            "canonical names (<var>.pdf). Used by the eval-data-contract to produce "
            "plots/spectra/<var>.pdf alongside the data/ artifacts."
        ),
    )
    return p.parse_args()


def parse_weather_states(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def valid_prediction_files(pred_dir: Path) -> list[Path]:
    files = sorted(pred_dir.glob("predictions_*.nc"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.nc files found in {pred_dir}")
    return files


def _fft_proxy_cl_from_unstructured(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    values: np.ndarray,
    *,
    lmax: int,
) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    lon = np.mod(np.asarray(lon_deg, dtype=np.float64).reshape(-1), 360.0)
    val = np.asarray(values, dtype=np.float64).reshape(-1)

    nlat = int(min(max(lmax + 1, 256), 2048))
    nlon = int(2 * nlat)

    lat_idx = np.clip(((lat + 90.0) / 180.0 * nlat).astype(np.int64), 0, nlat - 1)
    lon_idx = np.clip((lon / 360.0 * nlon).astype(np.int64), 0, nlon - 1)

    sums = np.zeros((nlat, nlon), dtype=np.float64)
    counts = np.zeros((nlat, nlon), dtype=np.int64)
    np.add.at(sums, (lat_idx, lon_idx), val)
    np.add.at(counts, (lat_idx, lon_idx), 1)

    grid = np.zeros((nlat, nlon), dtype=np.float64)
    valid = counts > 0
    if not np.any(valid):
        raise RuntimeError("No valid grid coverage for FFT spectra proxy.")

    grid[valid] = sums[valid] / counts[valid]
    grid_mean = float(np.mean(grid[valid]))
    grid[valid] = grid[valid] - grid_mean

    lat_centers = -90.0 + (np.arange(nlat, dtype=np.float64) + 0.5) * (180.0 / nlat)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat_centers)), 1e-6, None))[:, None]
    weighted = grid * weights

    spec2 = np.abs(np.fft.rfft2(weighted)) ** 2
    ky = np.fft.fftfreq(nlat)[:, None] * nlat
    kx = np.fft.rfftfreq(nlon)[None, :] * nlon
    kr = np.sqrt(kx**2 + ky**2)

    shell = np.floor(kr).astype(np.int64)
    keep = shell <= lmax
    power_sum = np.bincount(shell[keep].ravel(), weights=spec2[keep].ravel(), minlength=lmax + 1)
    power_count = np.bincount(shell[keep].ravel(), minlength=lmax + 1)
    cl = power_sum / np.maximum(power_count, 1)

    coverage = float(np.mean(valid))
    return cl / max(coverage, 1e-6)


def resolve_spectra_method(method: str):
    if method == "healpy":
        if _healpy_cl_from_unstructured is None:
            raise RuntimeError("Requested spectra-method=healpy but healpy path is unavailable.")
        return _healpy_cl_from_unstructured, "healpy"
    if method == "fft_proxy":
        return lambda lat, lon, val, nside, lmax: _fft_proxy_cl_from_unstructured(lat, lon, val, lmax=lmax), "fft_proxy"
    if _healpy_cl_from_unstructured is not None:
        return _healpy_cl_from_unstructured, "healpy"
    return lambda lat, lon, val, nside, lmax: _fft_proxy_cl_from_unstructured(lat, lon, val, lmax=lmax), "fft_proxy"


def finite_positive_mask(
    arr: np.ndarray,
    *,
    wavenumbers: np.ndarray | None = None,
    score_wavenumber_min_exclusive: float = SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE_DEFAULT,
) -> np.ndarray:
    ell = np.arange(arr.shape[0], dtype=np.float64) if wavenumbers is None else np.asarray(wavenumbers, dtype=np.float64)
    if ell.shape != arr.shape:
        raise ValueError(f"wavenumber length mismatch: curve={arr.shape[0]} wavenumbers={ell.shape[0]}")
    return np.isfinite(arr) & (arr > 0.0) & np.isfinite(ell) & (ell > score_wavenumber_min_exclusive)


def member_indices(ds: xr.Dataset) -> Iterable[int]:
    if "ensemble_member" not in ds.dims:
        return [0]
    return range(int(ds.sizes["ensemble_member"]))


def mean_curve(curves: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(np.stack(curves, axis=0), axis=0)


def select_member_array(da: xr.DataArray, member_idx: int) -> np.ndarray:
    member_da = da.isel(sample=0)
    if "ensemble_member" in member_da.dims:
        member_da = member_da.isel(ensemble_member=member_idx)
    return member_da.values.astype(np.float64)


def build_scope_curves(
    *,
    pred_dir: Path,
    files: list[Path],
    states: list[str],
    cl_from_unstructured,
    nside: int,
    lmax: int,
    member_aggregation: str,
    checkpoint_path_override: str,
    prediction_var: str = "y_pred",
) -> tuple[dict[str, dict[str, list[np.ndarray]]], dict[str, int], Path | None, list[str]]:
    scope_curves: dict[str, dict[str, list[np.ndarray]]] = {
        scope_name: {
            f"pred::{state}": []
            for state in states
        }
        for scope_name in SCOPE_SPECS
    }
    for scope_name in SCOPE_SPECS:
        for state in states:
            scope_curves[scope_name][f"truth::{state}"] = []
    for state in states:
        scope_curves["full_field"][f"input::{state}"] = []

    raw_member_curve_counts: dict[str, int] = {state: 0 for state in states}
    checkpoint_path: Path | None = None
    residualizer: CheckpointResidualInterpolator | None = None
    residualization_methods_used: set[str] = set()

    for file_path in files:
        with xr.open_dataset(file_path) as ds:
            if "x" not in ds:
                raise RuntimeError(f"Predictions file missing low-resolution input x required for residual spectra: {file_path}")
            if "y" not in ds:
                raise RuntimeError(f"Predictions file missing truth y required for spectra comparison: {file_path}")
            if prediction_var not in ds:
                raise RuntimeError(
                    f"Predictions file missing variable {prediction_var!r} required for spectra comparison: {file_path}"
                )

            weather_states = [str(value) for value in ds["weather_state"].values.tolist()]
            state_to_index = {state: idx for idx, state in enumerate(weather_states)}
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            if lat.ndim > 1:
                lat = lat[0]
            if lon.ndim > 1:
                lon = lon[0]

            file_has_x_interp = "x_interp" in ds
            if file_has_x_interp:
                residualization_methods_used.add("predictions_x_interp")
            else:
                residualization_methods_used.add("checkpoint_apply_interpolate_to_high_res_from_predictions_x")
                if checkpoint_path is None:
                    checkpoint_path = resolve_checkpoint_path(
                        pred_dir=pred_dir,
                        ds=ds,
                        explicit_path=checkpoint_path_override,
                    )
                    if checkpoint_path is None:
                        raise RuntimeError(
                            "Could not resolve checkpoint path for residual spectra reconstruction. "
                            "Set --checkpoint-path or ensure predictions metadata / EXPERIMENT_CONFIG.yaml carries checkpoint.path."
                        )
                    residualizer = CheckpointResidualInterpolator(checkpoint_path)

            per_file_curves: dict[str, dict[str, list[np.ndarray]]] = {
                scope_name: {f"pred::{state}": [] for state in states}
                for scope_name in SCOPE_SPECS
            }
            for scope_name in SCOPE_SPECS:
                for state in states:
                    per_file_curves[scope_name][f"truth::{state}"] = []
            for state in states:
                per_file_curves["full_field"][f"input::{state}"] = []

            for member_idx in member_indices(ds):
                x_member = select_member_array(ds["x"], member_idx)
                pred_member = select_member_array(ds[prediction_var], member_idx)
                truth_member = select_member_array(ds["y"], member_idx)
                if file_has_x_interp:
                    interp_member = select_member_array(ds["x_interp"], member_idx)
                else:
                    interp_member = residualizer.interpolate(x_member) if residualizer is not None else None

                if interp_member is None:
                    raise RuntimeError("Residual interpolator was not initialized.")
                if pred_member.ndim != 2 or truth_member.ndim != 2 or interp_member.ndim != 2:
                    raise ValueError(
                        "Expected per-member arrays with shape [grid_point, weather_state], got "
                        f"pred={pred_member.shape} truth={truth_member.shape} interp={interp_member.shape}"
                    )
                if pred_member.shape != truth_member.shape:
                    raise ValueError(
                        f"Prediction/truth shape mismatch in {file_path}: pred={pred_member.shape} truth={truth_member.shape}"
                    )
                if pred_member.shape != interp_member.shape:
                    raise ValueError(
                        "Interpolated low-resolution state shape does not match high-resolution outputs in "
                        f"{file_path}: interp={interp_member.shape} output={pred_member.shape}"
                    )

                for state in states:
                    state_index = state_to_index.get(state)
                    if state_index is None:
                        continue
                    pred_curve = cl_from_unstructured(lat, lon, pred_member[:, state_index], nside=nside, lmax=lmax)
                    truth_curve = cl_from_unstructured(lat, lon, truth_member[:, state_index], nside=nside, lmax=lmax)
                    input_curve = cl_from_unstructured(lat, lon, interp_member[:, state_index], nside=nside, lmax=lmax)
                    residual_pred_curve = cl_from_unstructured(
                        lat,
                        lon,
                        pred_member[:, state_index] - interp_member[:, state_index],
                        nside=nside,
                        lmax=lmax,
                    )
                    residual_truth_curve = cl_from_unstructured(
                        lat,
                        lon,
                        truth_member[:, state_index] - interp_member[:, state_index],
                        nside=nside,
                        lmax=lmax,
                    )
                    per_file_curves["full_field"][f"pred::{state}"].append(pred_curve)
                    per_file_curves["full_field"][f"truth::{state}"].append(truth_curve)
                    per_file_curves["full_field"][f"input::{state}"].append(input_curve)
                    per_file_curves["residual"][f"pred::{state}"].append(residual_pred_curve)
                    per_file_curves["residual"][f"truth::{state}"].append(residual_truth_curve)

            for state in states:
                member_count = len(per_file_curves["full_field"][f"pred::{state}"])
                raw_member_curve_counts[state] += member_count
                if member_count == 0:
                    continue
                for scope_name in SCOPE_SPECS:
                    pred_key = f"pred::{state}"
                    truth_key = f"truth::{state}"
                    pred_curves = per_file_curves[scope_name][pred_key]
                    truth_curves = per_file_curves[scope_name][truth_key]
                    if member_aggregation == "raw-members":
                        scope_curves[scope_name][pred_key].extend(pred_curves)
                        scope_curves[scope_name][truth_key].extend(truth_curves)
                    else:
                        scope_curves[scope_name][pred_key].append(mean_curve(pred_curves))
                        scope_curves[scope_name][truth_key].append(mean_curve(truth_curves))
                input_key = f"input::{state}"
                input_curves = per_file_curves["full_field"][input_key]
                if member_aggregation == "raw-members":
                    scope_curves["full_field"][input_key].extend(input_curves)
                else:
                    scope_curves["full_field"][input_key].append(mean_curve(input_curves))

    if not residualization_methods_used:
        raise RuntimeError("No residualization method was available while building spectra curves.")
    return scope_curves, raw_member_curve_counts, checkpoint_path, sorted(residualization_methods_used)


def plot_one_state(
    *,
    state: str,
    pred_cls: list[np.ndarray],
    truth_cls: list[np.ndarray],
    input_cls: list[np.ndarray] | None,
    out_pdf: Path,
    run_label: str,
    show_individual_curves: bool,
    aggregate_label: str,
    score_wavenumber_min_exclusive: float,
    scope: ScopeSpec,
) -> dict[str, float]:
    pred_mean = mean_curve(pred_cls)
    truth_mean = mean_curve(truth_cls)

    ell = np.arange(pred_mean.shape[0], dtype=np.float64)
    kp = finite_positive_mask(pred_mean, wavenumbers=ell, score_wavenumber_min_exclusive=0.0)
    kt = finite_positive_mask(truth_mean, wavenumbers=ell, score_wavenumber_min_exclusive=0.0)
    score_keep = finite_positive_mask(
        pred_mean,
        wavenumbers=ell,
        score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
    ) & finite_positive_mask(
        truth_mean,
        wavenumbers=ell,
        score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    if show_individual_curves:
        for arr in truth_cls:
            keep = finite_positive_mask(arr, wavenumbers=ell, score_wavenumber_min_exclusive=0.0)
            ax.plot(ell[keep], arr[keep], color="#888888", alpha=0.18, linewidth=0.8)
        for arr in pred_cls:
            keep = finite_positive_mask(arr, wavenumbers=ell, score_wavenumber_min_exclusive=0.0)
            ax.plot(ell[keep], arr[keep], color="#1f77b4", alpha=0.18, linewidth=0.8)

    if input_cls is not None and scope.curve_c_label is not None:
        input_mean = mean_curve(input_cls)
        ki = finite_positive_mask(input_mean, wavenumbers=ell, score_wavenumber_min_exclusive=0.0)
        ax.plot(ell[ki], input_mean[ki], color="#888888", linewidth=2.2, linestyle="--", label=scope.curve_c_label)

    ax.plot(ell[kt], truth_mean[kt], color="#333333", linewidth=2.2, label=scope.curve_b_label)
    ax.plot(ell[kp], pred_mean[kp], color="#1f77b4", linewidth=2.2, label=scope.curve_a_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber l")
    ax.set_ylabel("Power")
    ax.set_title(f"{run_label} | {state} | {scope.title} ({aggregate_label})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    if not np.any(score_keep):
        rel_l2 = float("nan")
    else:
        rel_l2 = float(np.linalg.norm((pred_mean - truth_mean)[score_keep]) / max(np.linalg.norm(truth_mean[score_keep]), 1e-12))
    return {"relative_l2_mean_curve": rel_l2}


def _build_spectra_artifact_summaries(
    *,
    pred_dir: Path,
    out_dir: Path,
    run_label: str,
    states: list[str],
    nside: int,
    lmax: int,
    spectra_method: str,
    member_aggregation: str,
    show_individual_curves: bool,
    score_wavenumber_min_exclusive: float,
    checkpoint_path_override: str = "",
    prediction_var: str = "y_pred",
) -> tuple[dict[str, object], dict[str, object]]:
    files = valid_prediction_files(pred_dir)
    cl_from_unstructured, spectra_method_used = resolve_spectra_method(spectra_method)
    scope_curves, raw_member_curve_counts, checkpoint_path, residualization_methods = build_scope_curves(
        pred_dir=pred_dir,
        files=files,
        states=states,
        cl_from_unstructured=cl_from_unstructured,
        nside=nside,
        lmax=lmax,
        member_aggregation=member_aggregation,
        checkpoint_path_override=checkpoint_path_override,
        prediction_var=prediction_var,
    )

    summary: dict[str, object] = {
        "run_label": run_label,
        "predictions_dir": str(pred_dir),
        "prediction_var": prediction_var,
        "num_files": len(files),
        "nside": nside,
        "lmax": lmax,
        "score_wavenumber_min_exclusive": score_wavenumber_min_exclusive,
        "spectra_method_requested": spectra_method,
        "spectra_method_used": spectra_method_used,
        "member_aggregation": member_aggregation,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "residualization": {
            "method": residualization_methods[0] if len(residualization_methods) == 1 else "mixed",
            "methods": residualization_methods,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        },
        "scopes": list(SCOPE_SPECS),
        "weather_states": {},
    }
    curve_summary: dict[str, object] = {
        "run_label": run_label,
        "predictions_dir": str(pred_dir),
        "spectra_method_used": spectra_method_used,
        "score_wavenumber_min_exclusive": score_wavenumber_min_exclusive,
        "weather_states": {},
    }
    aggregate_label = f"{len(files)}-file aggregate"

    for state in states:
        full_pred = scope_curves["full_field"][f"pred::{state}"]
        full_truth = scope_curves["full_field"][f"truth::{state}"]
        residual_pred = scope_curves["residual"][f"pred::{state}"]
        residual_truth = scope_curves["residual"][f"truth::{state}"]

        if not full_pred or not full_truth:
            summary["weather_states"][state] = {
                "status": "missing",
                "scopes": {
                    scope_name: {"status": "missing"}
                    for scope_name in SCOPE_SPECS
                },
            }
            curve_summary["weather_states"][state] = {
                "status": "missing",
                "scopes": {scope_name: {"status": "missing"} for scope_name in SCOPE_SPECS},
            }
            continue
        if not residual_pred or not residual_truth:
            raise RuntimeError(f"Residual spectra missing for weather state {state}; refusing to write a full-field-only summary.")

        scope_summaries: dict[str, object] = {}
        curve_summary["weather_states"][state] = {"status": "ok", "scopes": {}}
        for scope_name, spec in SCOPE_SPECS.items():
            pred_curves = scope_curves[scope_name][f"pred::{state}"]
            truth_curves = scope_curves[scope_name][f"truth::{state}"]
            input_curves = scope_curves["full_field"].get(f"input::{state}") if scope_name == "full_field" else None
            out_pdf = out_dir / spec.pdf_name.format(state=state)
            metrics = plot_one_state(
                state=state,
                pred_cls=pred_curves,
                truth_cls=truth_curves,
                input_cls=input_curves or None,
                out_pdf=out_pdf,
                run_label=run_label,
                show_individual_curves=show_individual_curves,
                aggregate_label=aggregate_label,
                score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
                scope=spec,
            )
            pred_mean = mean_curve(pred_curves)
            truth_mean = mean_curve(truth_curves)
            wavenumbers = np.arange(pred_mean.shape[0], dtype=np.float64)
            scope_summaries[scope_name] = {
                "status": "ok",
                "n_curves": len(pred_curves),
                "n_member_curves": raw_member_curve_counts[state],
                "pdf": str(out_pdf),
                **metrics,
            }
            curve_summary["weather_states"][state]["scopes"][scope_name] = {
                "status": "ok",
                "n_curves": len(pred_curves),
                "wavenumbers": wavenumbers.tolist(),
                "prediction_mean": pred_mean.tolist(),
                "truth_mean": truth_mean.tolist(),
                "pdf": str(out_pdf),
                "relative_l2_mean_curve": metrics["relative_l2_mean_curve"],
            }

        full_summary = dict(scope_summaries["full_field"])
        summary["weather_states"][state] = {
            "status": "ok",
            "n_curves": full_summary["n_curves"],
            "n_member_curves": full_summary["n_member_curves"],
            "pdf": full_summary["pdf"],
            "relative_l2_mean_curve": full_summary["relative_l2_mean_curve"],
            "scopes": scope_summaries,
        }

    return summary, curve_summary


class SpectraArtifactsResult(dict):
    def __init__(self, summary: dict[str, object], curve_summary: dict[str, object]):
        super().__init__(summary)
        self.curve_summary = curve_summary

    def __iter__(self):
        yield self
        yield self.curve_summary


def build_spectra_artifacts(
    *,
    pred_dir: Path,
    out_dir: Path,
    run_label: str,
    states: list[str],
    nside: int,
    lmax: int,
    spectra_method: str,
    member_aggregation: str,
    show_individual_curves: bool,
    score_wavenumber_min_exclusive: float,
    checkpoint_path_override: str = "",
    prediction_var: str = "y_pred",
) -> SpectraArtifactsResult:
    summary, curve_summary = _build_spectra_artifact_summaries(
        pred_dir=pred_dir,
        out_dir=out_dir,
        run_label=run_label,
        states=states,
        nside=nside,
        lmax=lmax,
        spectra_method=spectra_method,
        member_aggregation=member_aggregation,
        show_individual_curves=show_individual_curves,
        score_wavenumber_min_exclusive=score_wavenumber_min_exclusive,
        checkpoint_path_override=checkpoint_path_override,
        prediction_var=prediction_var,
    )
    return SpectraArtifactsResult(summary, curve_summary)


def _merge_pdfs_simple(source_pdfs: list[Path], target: Path) -> None:
    """Concatenate PDF files page by page using PyPDF2/pypdf when available, else copy first."""
    try:
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        for src in source_pdfs:
            reader = PdfReader(str(src))
            for page in reader.pages:
                writer.add_page(page)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            writer.write(f)
    except ImportError:
        import subprocess
        import shutil

        gs_path = shutil.which("gs")
        if gs_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    gs_path,
                    "-dBATCH",
                    "-dNOPAUSE",
                    "-q",
                    "-sDEVICE=pdfwrite",
                    f"-sOutputFile={target}",
                ]
                + [str(p) for p in source_pdfs],
                check=True,
            )
        else:
            raise RuntimeError(
                "Neither pypdf nor ghostscript (gs) available to merge PDFs. "
                "Install pypdf or ensure gs is on PATH."
            )


def build_consolidated_spectra_pdf_from_existing(
    *,
    out_dir: Path,
    consolidated_pdf_path: Path,
    states: list[str],
) -> Path:
    """Merge all per-variable per-scope spectra PDFs into one consolidated multi-page PDF."""
    per_variable_pdfs: list[Path] = []
    for state in states:
        for scope_name, spec in SCOPE_SPECS.items():
            candidate = out_dir / spec.pdf_name.format(state=state)
            if candidate.exists():
                per_variable_pdfs.append(candidate)

    if not per_variable_pdfs:
        raise FileNotFoundError(f"No per-variable spectra PDFs found in {out_dir}")

    _merge_pdfs_simple(per_variable_pdfs, consolidated_pdf_path)
    print(f"Wrote consolidated spectra PDF ({len(per_variable_pdfs)} pages): {consolidated_pdf_path}")
    return consolidated_pdf_path


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.predictions_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    states = parse_weather_states(args.weather_states)

    summary, curve_summary = build_spectra_artifacts(
        pred_dir=pred_dir,
        out_dir=out_dir,
        run_label=args.run_label,
        states=states,
        nside=args.nside,
        lmax=args.lmax,
        spectra_method=args.spectra_method,
        member_aggregation=args.member_aggregation,
        show_individual_curves=args.show_individual_curves,
        score_wavenumber_min_exclusive=args.score_wavenumber_min_exclusive,
        checkpoint_path_override=args.checkpoint_path,
        prediction_var=args.prediction_var,
    )

    out_json = out_dir / "spectra_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote spectra summary: {out_json}")

    # Canonical name: curves.json (eval-data-contract)
    out_curves_canonical = out_dir / "curves.json"
    out_curves_canonical.write_text(json.dumps(curve_summary, indent=2), encoding="utf-8")
    print(f"Wrote spectra curves (canonical): {out_curves_canonical}")

    # Legacy name for backward compat
    out_curves_legacy = out_dir / "spectra_curve_summary.json"
    out_curves_legacy.write_text(json.dumps(curve_summary, indent=2), encoding="utf-8")
    print(f"Wrote spectra curve summary (legacy): {out_curves_legacy}")

    # Canonical summary.json alias
    summary_alias = out_dir / "summary.json"
    if not summary_alias.exists():
        summary_alias.symlink_to("spectra_summary.json")

    # --plots-dir: copy per-variable PDFs before consolidation removes them.
    if args.plots_dir:
        plots_dir = Path(args.plots_dir).expanduser().resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for state in states:
            for scope_name, spec in SCOPE_SPECS.items():
                src_pdf = out_dir / spec.pdf_name.format(state=state)
                if src_pdf.exists():
                    if scope_name == "full_field":
                        dst_pdf = plots_dir / f"{state}.pdf"
                    else:
                        dst_pdf = plots_dir / f"{state}_{scope_name}.pdf"
                    shutil.copy2(src_pdf, dst_pdf)
        print(f"Copied per-variable PDFs to: {plots_dir}")

    # Always build the consolidated all_spectra.pdf, then remove individual per-state PDFs.
    # Only the consolidated file is kept in out_dir.
    consolidated_path = (
        Path(args.consolidated_pdf).expanduser().resolve()
        if args.consolidated_pdf
        else out_dir / "all_spectra_proxy.pdf"
    )
    try:
        build_consolidated_spectra_pdf_from_existing(
            out_dir=out_dir,
            consolidated_pdf_path=consolidated_path,
            states=states,
        )
        # Remove all individual per-state PDFs — only the consolidated file should remain.
        for individual in sorted(out_dir.glob("spectra_*.pdf")):
            if individual.resolve() != consolidated_path.resolve():
                individual.unlink()
    except Exception as exc:
        print(f"[WARN] PDF consolidation failed (non-fatal): {exc}", flush=True)
        print(
            f"[WARN] Re-run to generate PDF: "
            f"python spectra_plot_pdf.py --spectra-dir {out_dir} --out-pdf {consolidated_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
