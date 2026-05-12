from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval._backends.spectra.calibrate_fast_spectra_proxy import cl_from_unstructured


SCOPE_LABELS = {
    "full_field": "full field",
    "residual": "residual",
}
SCOPE_BASENAMES = {
    "full_field": "spectra_sigma_curve",
    "residual": "spectra_sigma_curve_residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure spectra discrepancy vs target for each intermediate step in both full-field and residual space."
    )
    parser.add_argument(
        "--intermediate-nc",
        default="/home/ecm5702/scratch/eval/j2hh_old_intermediate_idalia_20260304/j2hh_idalia_mem06_step048h__intermediate_cached.nc",
        help="Cached intermediate NetCDF from the TARGET run.",
    )
    parser.add_argument("--param", default="2t", help="Weather state (e.g., 2t, 10u, msl).")
    parser.add_argument(
        "--target-var",
        default="y",
        choices=["y", "y_pred"],
        help="Field used as the spectral target.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the NetCDF to analyze.",
    )
    parser.add_argument(
        "--ensemble-index",
        type=int,
        default=0,
        help="Ensemble member in the NetCDF to analyze.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="Healpix nside for the spherical transform.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=319,
        help="Maximum spherical harmonic degree.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/j2hh_old_intermediate_idalia_20260304/spectra_sigma_curve_20260305",
        help="Directory where the curve/table/README will be written.",
    )
    return parser.parse_args()


def build_sigma_schedule(num_steps: int, sigma_max: float, sigma_min: float, rho: float) -> Sequence[float]:
    if num_steps <= 1:
        return [sigma_min]
    sigma_max_r = sigma_max ** (1.0 / rho)
    sigma_min_r = sigma_min ** (1.0 / rho)
    schedule = []
    for step in range(num_steps):
        frac = step / (num_steps - 1)
        sigma_r = sigma_max_r + frac * (sigma_min_r - sigma_max_r)
        schedule.append(float(sigma_r ** rho))
    return schedule


def compute_cl_for_field(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    nside: int,
    lmax: int,
) -> np.ndarray:
    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return cl_from_unstructured(lat, lon, clean_values, nside=nside, lmax=lmax)


def format_fixed_width_table(records: Sequence[dict]) -> str:
    header = "step  sigma    mean_abs_log10  rmse    relative_l2"
    lines = [header]
    for rec in records:
        lines.append(
            f"{rec['step']:>4d}  "
            f"{rec['sigma']:>7.2f}  "
            f"{rec['mean_abs_log10']:>14.6f}  "
            f"{rec['rmse']:>6.4f}  "
            f"{rec['relative_l2']:>11.6f}"
        )
    return "\n".join(lines)


def _select_member_array(da: xr.DataArray, *, sample_index: int, ensemble_index: int) -> np.ndarray:
    selected = da.isel(sample=sample_index)
    if "ensemble_member" in selected.dims:
        selected = selected.isel(ensemble_member=ensemble_index)
    return np.asarray(selected.values, dtype=np.float64)


def evaluate_sigma_curves(
    ds: xr.Dataset,
    *,
    param: str,
    target_var: str,
    sample_index: int,
    ensemble_index: int,
    nside: int,
    lmax: int,
) -> dict[str, object]:
    lat = np.asarray(ds["lat_hres"], dtype=np.float64)
    lon = np.asarray(ds["lon_hres"], dtype=np.float64)
    weather_states = ds["weather_state"].values.astype(str)
    if param not in weather_states:
        raise ValueError(f"{param} not found in weather_state coordinates.")
    if "x_interp" not in ds:
        raise ValueError("Intermediate dataset is missing x_interp required for residual spectra.")

    weather_index = int(np.where(weather_states == param)[0][0])
    x_interp_values = _select_member_array(
        ds["x_interp"],
        sample_index=sample_index,
        ensemble_index=ensemble_index,
    )[:, weather_index]

    target_values = _select_member_array(
        ds[target_var],
        sample_index=sample_index,
        ensemble_index=ensemble_index,
    )[:, weather_index]
    target_cl_full = compute_cl_for_field(lat, lon, target_values.ravel(), nside, lmax)
    target_cl_residual = compute_cl_for_field(lat, lon, (target_values - x_interp_values).ravel(), nside, lmax)

    inter = ds["inter_state"]
    num_steps = int(inter.sizes["sampling_step"])
    sampling_config = json.loads(ds.attrs.get("sampling_config_json", "{}") or "{}")
    schedule = build_sigma_schedule(
        num_steps,
        float(sampling_config.get("sigma_max", 1000.0)),
        float(sampling_config.get("sigma_min", 0.03)),
        float(sampling_config.get("rho", 7.0)),
    )

    records_by_scope: dict[str, list[dict[str, float]]] = {scope: [] for scope in SCOPE_LABELS}
    eps = 1e-30
    for step in range(num_steps):
        values = _select_member_array(
            inter.isel(sampling_step=step),
            sample_index=sample_index,
            ensemble_index=ensemble_index,
        )[:, weather_index]

        comparisons = {
            "full_field": (values, target_cl_full),
            "residual": (values - x_interp_values, target_cl_residual),
        }
        for scope_name, (field_values, target_cl) in comparisons.items():
            cl = compute_cl_for_field(lat, lon, field_values.ravel(), nside, lmax)
            n = min(len(cl), len(target_cl))
            inter_slice = cl[:n]
            target_slice = target_cl[:n]
            ratio = np.maximum(inter_slice, eps) / np.maximum(target_slice, eps)
            mean_abs_log10 = float(np.mean(np.abs(np.log10(ratio))))
            rmse = float(np.sqrt(np.mean((inter_slice - target_slice) ** 2)))
            relative_l2 = float(np.linalg.norm(inter_slice - target_slice) / max(np.linalg.norm(target_slice), eps))
            records_by_scope[scope_name].append(
                {
                    "step": step,
                    "sigma": schedule[step] if step < len(schedule) else 0.0,
                    "mean_abs_log10": mean_abs_log10,
                    "rmse": rmse,
                    "relative_l2": relative_l2,
                }
            )

    return {
        "param": param,
        "target_var": target_var,
        "sigma_schedule": schedule,
        "records_by_scope": records_by_scope,
    }


def write_scope_outputs(
    *,
    output_dir: Path,
    param: str,
    target_var: str,
    sigma_schedule: Sequence[float],
    records_by_scope: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for scope_name, records in records_by_scope.items():
        base_name = SCOPE_BASENAMES[scope_name]
        table_path = output_dir / f"{base_name}.txt"
        table_path.write_text(format_fixed_width_table(records) + "\n", encoding="utf-8")

        plot_path = output_dir / f"{base_name}.pdf"
        png_path = output_dir / f"{base_name}.png"
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        sigmas = [rec["sigma"] for rec in records]
        mean_abs = [rec["mean_abs_log10"] for rec in records]
        ax.plot(sigmas, mean_abs, label="Mean |log10 ratio|", color="#1f77b4", marker="o", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlabel("Sigma")
        ax.set_ylabel("Mean |log10(inter / target)|")
        ax.set_title(f"{param} | {SCOPE_LABELS[scope_name]} spectra vs {target_var}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        ax2 = ax.twinx()
        rmse = [rec["rmse"] for rec in records]
        ax2.plot(sigmas, rmse, label="RMSE", color="#d62728", marker="s", linewidth=1.2)
        ax2.set_ylabel("RMSE (power spectrum)")
        ax2.set_yscale("linear")
        ax2.tick_params(axis="y", colors="#d62728")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=240, bbox_inches="tight")
        fig.savefig(png_path, dpi=240, bbox_inches="tight")
        plt.close(fig)

        outputs[scope_name] = {
            "table": str(table_path),
            "plot": str(plot_path),
            "plot_png": str(png_path),
        }

    metadata = {
        "param": param,
        "target_var": target_var,
        "sigma_schedule": list(sigma_schedule),
        "table": outputs["full_field"]["table"],
        "plot": outputs["full_field"]["plot"],
        "plot_png": outputs["full_field"]["plot_png"],
        "scopes": {
            scope_name: {
                **outputs[scope_name],
                "label": SCOPE_LABELS[scope_name],
            }
            for scope_name in outputs
        },
    }
    metadata_path = output_dir / "spectra_sigma_curve.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outputs | {"metadata": {"path": str(metadata_path)}}


def main() -> None:
    args = parse_args()
    ds = xr.open_dataset(args.intermediate_nc)
    try:
        result = evaluate_sigma_curves(
            ds,
            param=args.param,
            target_var=args.target_var,
            sample_index=args.sample_index,
            ensemble_index=args.ensemble_index,
            nside=args.nside,
            lmax=args.lmax,
        )
    finally:
        ds.close()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_scope_outputs(
        output_dir=out_dir,
        param=args.param,
        target_var=args.target_var,
        sigma_schedule=result["sigma_schedule"],
        records_by_scope=result["records_by_scope"],
    )

    for scope_name in ("full_field", "residual"):
        scope_outputs = outputs[scope_name]
        print(f"Wrote {scope_name} table -> {scope_outputs['table']}")
        print(f"Wrote {scope_name} plot -> {scope_outputs['plot']}")
        print(f"Wrote {scope_name} PNG -> {scope_outputs['plot_png']}")
    print(f"Wrote metadata -> {outputs['metadata']['path']}")


if __name__ == "__main__":
    main()
