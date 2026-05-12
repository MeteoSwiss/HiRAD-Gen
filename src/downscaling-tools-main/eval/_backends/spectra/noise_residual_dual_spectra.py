from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import xarray as xr
from eval.paths import reference_spectra_dir
from eccodes import (
    codes_clone,
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
    codes_set_values,
    codes_write,
)


TARGET_VARIABLES = ["10u", "10v", "2t", "t_850"]
NOISE_STDS = np.logspace(-1, 7, 9)


@dataclass(frozen=True)
class VariableConfig:
    name: str
    template_dir: str
    short_name: str
    levelist: int | None


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    high_res_label: str
    low_dataset: str
    high_dataset: str
    matrix_file: str
    residual_stats_file: str
    template_root: str


VARIABLE_CONFIGS: dict[str, VariableConfig] = {
    "10u": VariableConfig(name="10u", template_dir="10u_sfc", short_name="10u", levelist=None),
    "10v": VariableConfig(name="10v", template_dir="10v_sfc", short_name="10v", levelist=None),
    "2t": VariableConfig(name="2t", template_dir="2t_sfc", short_name="2t", levelist=None),
    "t_850": VariableConfig(name="t_850", template_dir="t_850", short_name="t", levelist=850),
}


CASE_CONFIGS: list[CaseConfig] = [
    CaseConfig(
        case_id="o320_from_o96",
        high_res_label="O320",
        low_dataset="/home/mlx/ai-ml/datasets/downscaling-od-cf-eefh-0001-mars-o96-2003-2023-12h-v3.zarr",
        high_dataset="/home/mlx/ai-ml/datasets/downscaling-od-cf-enfh-0001-mars-o320-2003-2023-12h-v3.zarr",
        matrix_file="/home/ecm5702/hpcperm/data/inter_mat/interpol_O96_to_O320_linear.mat.npz",
        residual_stats_file="/home/ecm5702/hpcperm/data/residuals_statistics/o320_dict_0_72.npy",
        template_root=str(reference_spectra_dir("enfo_o320")),
    ),
    CaseConfig(
        case_id="o1280_from_o320",
        high_res_label="O1280",
        low_dataset="/home/mlx/ai-ml/datasets/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-12h-v3.zarr",
        high_dataset="/home/mlx/ai-ml/datasets/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-12h-v3.zarr",
        matrix_file="/home/ecm5702/hpcperm/data/inter_mat/interpol_O320_to_O1280_linear.mat.npz",
        residual_stats_file="/home/ecm5702/hpcperm/data/residuals_statistics/o1280_dict_0_72.npy",
        template_root=str(reference_spectra_dir("enfo_o1280")),
    ),
    CaseConfig(
        case_id="o2560_from_o1280",
        high_res_label="O2560",
        low_dataset="/home/mlx/ai-ml/datasets/downscaling-od-fc-oper-0001-mars-o1280-2023-2024-24h-v1.zarr",
        high_dataset="/home/mlx/ai-ml/datasets/downscaling-rd-fc-oper-i4ql-mars-o2560-2023-2024-24h-v1.zarr",
        matrix_file="/home/ecm5702/hpcperm/data/inter_mat/interpol_O1280_to_O2560_linear.mat.npz",
        residual_stats_file="/home/ecm5702/hpcperm/data/residuals_statistics/o2560_dict_6_72.npy",
        template_root=str(reference_spectra_dir("destine_o2560_i4ql")),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual Gaussian-noise spectra study with two methods: "
            "gp2sp (gptosp+metview) and healpy anafast."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/noise_residual_spectra_20260305",
        help="Persistent output directory.",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sampled times per case.")
    parser.add_argument(
        "--truncation",
        type=int,
        default=319,
        help="gp2sp truncation (T319 is memory-safe on login sessions).",
    )
    parser.add_argument(
        "--healpy-nside",
        type=int,
        default=512,
        help="HEALPix nside for alternate spectra method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260305,
        help="Base seed for Gaussian noise generation.",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Comma-separated case ids to run (default: all).",
    )
    parser.add_argument(
        "--variables",
        default="",
        help="Comma-separated variables to run (default: 10u,10v,2t,t_850).",
    )
    parser.add_argument(
        "--noise-stds",
        default="",
        help="Comma-separated noise std overrides (default: 1e-1..1e7 logspace, 9 points).",
    )
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _variables_list(ds: xr.Dataset) -> list[str]:
    raw = ds.attrs.get("variables")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        return raw
    raise ValueError("Could not parse dataset variable list from zarr attrs['variables'].")


def _sample_indices(nt: int, n: int) -> list[int]:
    if n <= 0:
        raise ValueError("num-samples must be positive")
    idx = np.linspace(0, nt - 1, num=min(n, nt), dtype=np.int64)
    return sorted({int(x) for x in idx})


def _load_residual_stats(path: str) -> dict[str, dict[str, float]]:
    obj = np.load(path, allow_pickle=True).item()
    if not isinstance(obj, dict) or "mean" not in obj or "stdev" not in obj:
        raise ValueError(f"Unexpected residual statistics format: {path}")
    return obj


def _load_sparse_matrix(path: str) -> sp.csr_matrix:
    matrix = sp.load_npz(path).tocsr()
    data = matrix.data.astype(np.float32, copy=False)
    indices = matrix.indices.astype(np.int32, copy=False)
    indptr = matrix.indptr.astype(np.int32, copy=False)
    return sp.csr_matrix((data, indices, indptr), shape=matrix.shape, dtype=np.float32)


def _find_template(template_root: str, var_cfg: VariableConfig) -> Path:
    candidates = sorted((Path(template_root) / var_cfg.template_dir).glob("*_nopoles.grb"))
    if not candidates:
        raise FileNotFoundError(f"No template GRIB found in {Path(template_root) / var_cfg.template_dir}")
    return candidates[0]


def _compute_nopole_mask(lat_deg: np.ndarray, template_path: Path) -> np.ndarray:
    """Boolean mask (True = keep) that trims zarr pole points to match the GRIB template grid."""
    with template_path.open("rb") as f_in:
        gid = codes_grib_new_from_file(f_in)
        nvals = int(codes_get(gid, "numberOfValues"))
        lat_first = float(codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        codes_release(gid)
    mask = np.abs(lat_deg) <= lat_first + 0.001
    if int(mask.sum()) != nvals:
        raise ValueError(
            f"Pole-mask count {mask.sum()} != template numberOfValues {nvals}. "
            f"lat_first={lat_first}, zarr cells={len(lat_deg)}"
        )
    return mask


def _write_grib_from_template(template_path: Path, out_path: Path, values: np.ndarray) -> None:
    with template_path.open("rb") as f_in:
        gid = codes_grib_new_from_file(f_in)
        if gid is None:
            raise RuntimeError(f"Could not read GRIB template: {template_path}")
        clone = codes_clone(gid)
        codes_release(gid)

    try:
        codes_set_values(clone, np.asarray(values, dtype=np.float64))
        with out_path.open("wb") as f_out:
            codes_write(clone, f_out)
    finally:
        codes_release(clone)


def _run_gp2sp(grib_in: Path, sh_out: Path, truncation: int) -> None:
    cmd = (
        "module load ifs >/dev/null 2>&1 && "
        f"gptosp.ser -l -T {truncation} -g {grib_in} -S {sh_out}"
    )
    completed = subprocess.run(
        ["bash", "-lc", cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"gptosp failed for {grib_in}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _spectral_power_from_sh_grib(sh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with sh_path.open("rb") as f_in:
        gid = codes_grib_new_from_file(f_in)
        if gid is None:
            raise RuntimeError(f"Could not read spectral GRIB: {sh_path}")
        try:
            trunc = int(codes_get(gid, "J"))
            values = np.asarray(codes_get_array(gid, "values"), dtype=np.float64)
        finally:
            codes_release(gid)

    ncoef = (trunc + 1) * (trunc + 2) // 2
    if values.size == 2 * ncoef:
        coeff = values.reshape(ncoef, 2)
    elif values.size == ncoef:
        coeff = np.column_stack([values, np.zeros_like(values)])
    else:
        raise ValueError(
            f"Unexpected spectral coefficient size {values.size} for truncation {trunc} in {sh_path}"
        )

    power = np.zeros(trunc + 1, dtype=np.float64)
    pos = 0
    for m in range(trunc + 1):
        count = trunc - m + 1
        block = coeff[pos : pos + count]
        amp2 = block[:, 0] * block[:, 0] + block[:, 1] * block[:, 1]
        factor = 1.0 if m == 0 else 2.0
        amp2 *= factor
        power[m : trunc + 1] += amp2
        pos += count

    wvn = np.arange(trunc + 1, dtype=np.float64)
    return wvn, power


def _healpy_setup(
    lat_deg: np.ndarray, lon_deg: np.ndarray, nside: int
) -> tuple[np.ndarray, np.ndarray, int]:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    valid_coord = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid_coord):
        raise ValueError("No finite coordinates available for HEALPix setup.")

    valid_idx = np.nonzero(valid_coord)[0].astype(np.int64)
    theta = np.deg2rad(90.0 - lat[valid_coord])
    phi = np.deg2rad(np.mod(lon[valid_coord], 360.0))
    theta = np.clip(theta, 0.0, np.pi)
    phi = np.mod(phi, 2.0 * np.pi)

    pix = hp.ang2pix(nside, theta, phi, nest=False).astype(np.int32)
    npix = hp.nside2npix(nside)
    return valid_idx, pix, npix


def _healpy_spectrum(
    values: np.ndarray,
    valid_idx: np.ndarray,
    pix: np.ndarray,
    npix: int,
    lmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)[valid_idx]
    finite_vals = np.isfinite(vals)
    if not np.any(finite_vals):
        ell = np.arange(lmax + 1, dtype=np.float64)
        return ell, np.full(lmax + 1, np.nan, dtype=np.float64)

    vals = vals[finite_vals]
    pix_loc = pix[finite_vals]
    sums = np.bincount(pix_loc, weights=vals, minlength=npix)
    counts = np.bincount(pix_loc, minlength=npix).astype(np.int64)
    valid = counts > 0
    m = np.zeros(npix, dtype=np.float64)
    m[valid] = sums[valid] / counts[valid]
    m[valid] -= np.mean(m[valid])
    cl = hp.anafast(m, lmax=lmax)
    coverage = float(np.count_nonzero(valid)) / float(npix)
    if coverage > 0.0:
        cl = cl / coverage
    ell = np.arange(len(cl), dtype=np.float64)
    return ell, cl


def _dominance(noisy_y: np.ndarray, clean_y: np.ndarray) -> tuple[float, float]:
    n = min(len(noisy_y), len(clean_y))
    eps = 1e-30
    ratio = noisy_y[:n] / np.maximum(clean_y[:n], eps)
    keep = np.isfinite(ratio) & (ratio > 0.0)
    if keep.sum() <= 3:
        return float("nan"), float("nan")
    ratio = ratio[keep][3:]
    if ratio.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(ratio > 1.0)), float(np.median(ratio))


def _threshold_from_series(levels: list[float], fractions: list[float], target: float) -> float | None:
    for s, f in zip(levels, fractions):
        if math.isfinite(f) and f >= target:
            return s
    return None


def _stable_seed(base_seed: int, case_id: str, var: str, std: float) -> int:
    payload = f"{base_seed}|{case_id}|{var}|{std:.16e}".encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return int(digest, 16)


def _plot_case_method(
    case_id: str,
    method: str,
    spectra: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    noise_stds: np.ndarray,
    out_path: Path,
) -> None:
    plot_vars = list(spectra.keys())
    ncols = 2
    nrows = max(1, (len(plot_vars) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.1 * nrows), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    axes = np.atleast_1d(axes).ravel()

    for iax, var in enumerate(plot_vars):
        ax = axes[iax]
        cur = spectra[var]
        x_clean, y_clean = cur[0.0]
        ax.plot(x_clean[3:], y_clean[3:], color="black", linewidth=2.2, label="clean")
        for i, std in enumerate(noise_stds):
            x, y = cur[float(std)]
            col = cmap(i / max(1, len(noise_stds) - 1))
            ax.plot(x[3:], y[3:], color=col, linewidth=1.05, alpha=0.95, label=f"{std:.1e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(color="0.75", linestyle="--", linewidth=0.35, alpha=0.8)
        ax.set_title(var, fontsize=11, pad=8)
        ax.set_xlabel("Wavenumber", fontsize=9)
        ax.set_ylabel("Power", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        if iax == 0:
            ax.legend(
                fontsize=7.3,
                frameon=False,
                ncol=2,
                title="noise std",
                title_fontsize=8,
                loc="best",
            )

    for extra in range(len(plot_vars), len(axes)):
        axes[extra].set_visible(False)
    fig.suptitle(f"{case_id} - {method} spectra (clean + gaussian noise)", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dominance_heatmap(
    method: str,
    dominance_rows: list[dict[str, Any]],
    noise_stds: np.ndarray,
    out_path: Path,
) -> None:
    labels = [f"{row['case']}:{row['var']}" for row in dominance_rows]
    data = np.array([row["dominance_by_std"] for row in dominance_rows], dtype=np.float64)
    fig_h = max(6.0, 0.35 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h), constrained_layout=True)

    im = ax.imshow(data, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(f"Dominance fraction by noise std ({method})", fontsize=12, pad=10)
    ax.set_xlabel("Gaussian noise std", fontsize=10)
    ax.set_ylabel("Case:Variable", fontsize=10)
    ax.set_xticks(np.arange(len(noise_stds)))
    ax.set_xticklabels([f"{s:.1e}" for s in noise_stds], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label("Fraction of scales where noisy > clean", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_pretty_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'case':<19} {'var':<7} {'method':<8} "
        f"{'hide50_std':>11} {'hide90_std':>11} {'dom@1e7':>10} {'med_ratio@1e7':>14}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        hide50 = "na" if r["hide50_std"] is None else f"{r['hide50_std']:.1e}"
        hide90 = "na" if r["hide90_std"] is None else f"{r['hide90_std']:.1e}"
        dom = "na" if not math.isfinite(r["dom_at_1e7"]) else f"{r['dom_at_1e7']:.3f}"
        med = "na" if not math.isfinite(r["median_ratio_at_1e7"]) else f"{r['median_ratio_at_1e7']:.3e}"
        lines.append(
            f"{r['case']:<19} {r['var']:<7} {r['method']:<8} "
            f"{hide50:>11} {hide90:>11} {dom:>10} {med:>14}"
        )
    return "\n".join(lines) + "\n"


def _format_full_dominance_table(rows: list[dict[str, Any]], noise_stds: np.ndarray) -> str:
    std_cols = " ".join([f"{s:>9.1e}" for s in noise_stds])
    header = f"{'case':<19} {'var':<7} {'method':<8} {std_cols}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        vals = " ".join(
            [f"{v:>9.3f}" if math.isfinite(v) else f"{'na':>9}" for v in r["dominance_by_std"]]
        )
        lines.append(f"{r['case']:<19} {r['var']:<7} {r['method']:<8} {vals}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    selected_cases = set(_parse_csv(args.cases))
    selected_vars = _parse_csv(args.variables)
    vars_to_run = selected_vars if selected_vars else list(TARGET_VARIABLES)
    for v in vars_to_run:
        if v not in VARIABLE_CONFIGS:
            raise ValueError(f"Unsupported variable '{v}'. Allowed: {sorted(VARIABLE_CONFIGS)}")
    noise_stds = np.array(_parse_csv_floats(args.noise_stds), dtype=np.float64) if args.noise_stds else NOISE_STDS
    if noise_stds.size == 0:
        raise ValueError("At least one noise std is required.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work_gp2sp"
    work_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "noise_stds": [float(x) for x in noise_stds],
        "num_samples": int(args.num_samples),
        "truncation": int(args.truncation),
        "healpy_nside": int(args.healpy_nside),
        "seed": int(args.seed),
        "cases": {},
    }

    all_rows: list[dict[str, Any]] = []
    dominance_rows_by_method: dict[str, list[dict[str, Any]]] = {"gp2sp": [], "healpy": []}

    for case in CASE_CONFIGS:
        if selected_cases and case.case_id not in selected_cases:
            continue
        print(f"\n=== Processing case: {case.case_id}")
        ds_high = xr.open_zarr(case.high_dataset, consolidated=False)
        ds_low = xr.open_zarr(case.low_dataset, consolidated=False)
        try:
            high_vars = _variables_list(ds_high)
            low_vars = _variables_list(ds_low)

            nt = int(ds_high.sizes["time"])
            sample_idx = _sample_indices(nt, args.num_samples)
            dates = ds_high["dates"].isel(time=sample_idx).values.astype("datetime64[s]").astype(str).tolist()

            matrix = _load_sparse_matrix(case.matrix_file)
            if matrix.shape[0] != int(ds_high.sizes["cell"]) or matrix.shape[1] != int(ds_low.sizes["cell"]):
                raise ValueError(
                    f"Matrix shape {matrix.shape} does not match high/low cell counts "
                    f"{ds_high.sizes['cell']} / {ds_low.sizes['cell']}"
                )

            stats = _load_residual_stats(case.residual_stats_file)
            mean_stats = stats["mean"]
            stdev_stats = stats["stdev"]

            lat = ds_high["latitudes"].values
            lon = ds_high["longitudes"].values
            valid_idx, pix, npix = _healpy_setup(lat, lon, args.healpy_nside)
            # lat is kept for nopole mask computation per variable

            case_info: dict[str, Any] = {
                "high_dataset": case.high_dataset,
                "low_dataset": case.low_dataset,
                "matrix_file": case.matrix_file,
                "residual_stats_file": case.residual_stats_file,
                "sample_indices": sample_idx,
                "sample_dates": dates,
                "variables": {},
            }

            spectra_by_method: dict[str, dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]] = {
                "gp2sp": {},
                "healpy": {},
            }

            for var in vars_to_run:
                print(f"  - Variable: {var}")
                if var not in high_vars or var not in low_vars:
                    raise KeyError(f"Missing variable {var} in high or low dataset for case {case.case_id}")
                if var not in mean_stats or var not in stdev_stats:
                    raise KeyError(f"Missing variable {var} in residual stats file {case.residual_stats_file}")

                var_cfg = VARIABLE_CONFIGS[var]
                high_idx = high_vars.index(var)
                low_idx = low_vars.index(var)
                mean_val = float(mean_stats[var])
                std_val = float(stdev_stats[var])
                if std_val <= 0.0:
                    raise ValueError(f"Non-positive stdev for {var} in {case.residual_stats_file}")

                # Aggregate normalized residual over sampled times to limit cost while using 5 samples.
                accum: np.ndarray | None = None
                for ti in sample_idx:
                    high_field = (
                        ds_high["data"]
                        .isel(time=ti, variable=high_idx, ensemble=0)
                        .values.astype(np.float32, copy=False)
                    )
                    low_field = (
                        ds_low["data"]
                        .isel(time=ti, variable=low_idx, ensemble=0)
                        .values.astype(np.float32, copy=False)
                    )
                    interp_low = matrix.dot(low_field).astype(np.float32, copy=False)
                    residual = high_field - interp_low
                    normalized = (residual - mean_val) / std_val
                    if accum is None:
                        accum = np.zeros_like(normalized, dtype=np.float64)
                    accum += normalized.astype(np.float64, copy=False)

                assert accum is not None
                clean_field = (accum / len(sample_idx)).astype(np.float32)
                del accum

                template_path = _find_template(case.template_root, var_cfg)
                nopole_mask = _compute_nopole_mask(lat, template_path)
                var_gp2sp: dict[float, tuple[np.ndarray, np.ndarray]] = {}
                var_healpy: dict[float, tuple[np.ndarray, np.ndarray]] = {}
                dominance_gp2sp: list[float] = []
                dominance_healpy: list[float] = []
                median_gp2sp: list[float] = []
                median_healpy: list[float] = []

                # Clean spectra first.
                clean_grib = work_dir / f"{case.case_id}_{var}_clean.grb"
                clean_sh = work_dir / f"{case.case_id}_{var}_clean.grb_sh"
                _write_grib_from_template(template_path, clean_grib, clean_field[nopole_mask])
                _run_gp2sp(clean_grib, clean_sh, args.truncation)
                var_gp2sp[0.0] = _spectral_power_from_sh_grib(clean_sh)
                var_healpy[0.0] = _healpy_spectrum(clean_field, valid_idx, pix, npix, args.truncation)

                clean_gp2sp_y = var_gp2sp[0.0][1]
                clean_healpy_y = var_healpy[0.0][1]

                for std in noise_stds:
                    rng_seed = _stable_seed(args.seed, case.case_id, var, float(std))
                    rng = np.random.default_rng(rng_seed)
                    noise = rng.normal(0.0, float(std), size=clean_field.shape).astype(np.float32)
                    noisy_field = clean_field + noise

                    grib_path = work_dir / f"{case.case_id}_{var}_std{std:.1e}.grb"
                    sh_path = work_dir / f"{case.case_id}_{var}_std{std:.1e}.grb_sh"
                    _write_grib_from_template(template_path, grib_path, noisy_field[nopole_mask])
                    _run_gp2sp(grib_path, sh_path, args.truncation)
                    gp2sp_spec = _spectral_power_from_sh_grib(sh_path)
                    healpy_spec = _healpy_spectrum(noisy_field, valid_idx, pix, npix, args.truncation)

                    var_gp2sp[float(std)] = gp2sp_spec
                    var_healpy[float(std)] = healpy_spec

                    dom_gp, med_gp = _dominance(gp2sp_spec[1], clean_gp2sp_y)
                    dom_hp, med_hp = _dominance(healpy_spec[1], clean_healpy_y)
                    dominance_gp2sp.append(dom_gp)
                    dominance_healpy.append(dom_hp)
                    median_gp2sp.append(med_gp)
                    median_healpy.append(med_hp)

                spectra_by_method["gp2sp"][var] = var_gp2sp
                spectra_by_method["healpy"][var] = var_healpy

                hide50_gp = _threshold_from_series([float(x) for x in noise_stds], dominance_gp2sp, 0.5)
                hide90_gp = _threshold_from_series([float(x) for x in noise_stds], dominance_gp2sp, 0.9)
                hide50_hp = _threshold_from_series([float(x) for x in noise_stds], dominance_healpy, 0.5)
                hide90_hp = _threshold_from_series([float(x) for x in noise_stds], dominance_healpy, 0.9)

                all_rows.append(
                    {
                        "case": case.case_id,
                        "var": var,
                        "method": "gp2sp",
                        "hide50_std": hide50_gp,
                        "hide90_std": hide90_gp,
                        "dom_at_1e7": dominance_gp2sp[-1],
                        "median_ratio_at_1e7": median_gp2sp[-1],
                        "dominance_by_std": dominance_gp2sp,
                    }
                )
                all_rows.append(
                    {
                        "case": case.case_id,
                        "var": var,
                        "method": "healpy",
                        "hide50_std": hide50_hp,
                        "hide90_std": hide90_hp,
                        "dom_at_1e7": dominance_healpy[-1],
                        "median_ratio_at_1e7": median_healpy[-1],
                        "dominance_by_std": dominance_healpy,
                    }
                )

                dominance_rows_by_method["gp2sp"].append(
                    {
                        "case": case.case_id,
                        "var": var,
                        "dominance_by_std": dominance_gp2sp,
                    }
                )
                dominance_rows_by_method["healpy"].append(
                    {
                        "case": case.case_id,
                        "var": var,
                        "dominance_by_std": dominance_healpy,
                    }
                )

                case_info["variables"][var] = {
                    "mean": mean_val,
                    "stdev": std_val,
                    "template_path": str(template_path),
                    "hide50_std_gp2sp": hide50_gp,
                    "hide90_std_gp2sp": hide90_gp,
                    "hide50_std_healpy": hide50_hp,
                    "hide90_std_healpy": hide90_hp,
                }

            # Plot per-case per-method spectra.
            for method in ["gp2sp", "healpy"]:
                out_pdf = out_dir / f"spectra_{case.case_id}_{method}.pdf"
                _plot_case_method(case.case_id, method, spectra_by_method[method], noise_stds, out_pdf)

            summary["cases"][case.case_id] = case_info
        finally:
            ds_high.close()
            ds_low.close()
            del ds_high, ds_low
            gc.collect()

    # Global outputs.
    for method in ["gp2sp", "healpy"]:
        out_heat = out_dir / f"dominance_heatmap_{method}.pdf"
        _plot_dominance_heatmap(method, dominance_rows_by_method[method], noise_stds, out_heat)

    pretty = _format_pretty_table(all_rows)
    full = _format_full_dominance_table(all_rows, noise_stds)
    (out_dir / "NOISE_HIDE_THRESHOLDS_PRETTY.txt").write_text(pretty)
    (out_dir / "NOISE_DOMINANCE_BY_STD_PRETTY.txt").write_text(full)
    (out_dir / "noise_residual_spectra_summary.json").write_text(json.dumps(summary, indent=2))

    # Keep generated gp2sp intermediates for reproducibility only when needed.
    # Large intermediate files are removed by default to avoid accidental storage bloat.
    if work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
