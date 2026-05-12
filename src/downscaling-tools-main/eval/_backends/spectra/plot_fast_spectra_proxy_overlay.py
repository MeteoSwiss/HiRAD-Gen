from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval._backends.spectra.calibrate_fast_spectra_proxy import (
    apply_log_model,
    cl_from_unstructured,
    read_grib_latlon_values,
)


PARAM_CONFIGS = [
    {"param": "2t", "level": "sfc", "dir_name": "2t_sfc"},
    {"param": "10u", "level": "sfc", "dir_name": "10u_sfc"},
    {"param": "10v", "level": "sfc", "dir_name": "10v_sfc"},
    {"param": "sp", "level": "sfc", "dir_name": "sp_sfc"},
    {"param": "t", "level": "850", "dir_name": "t_850"},
    {"param": "z", "level": "500", "dir_name": "z_500"},
]
PARAM_TO_DIR = {(cfg["param"], cfg["level"]): cfg["dir_name"] for cfg in PARAM_CONFIGS}
AMPL_RE = re.compile(
    r"^ampl_(?P<date>\d{8})_(?P<step>\d+)_(?P<param>[a-z0-9]+)_(?P<level>[a-z0-9]+)_(?P<expid>[a-z0-9]+)_n(?P<number>\d+)\.npy$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot one fast spectra proxy overlay against reference spectra amplitudes."
    )
    p.add_argument(
        "--coefficients-json",
        default="/home/ecm5702/scratch/eval/spectra_proxy_calibration/20260303_fast_proxy_bestof_v2_v3/fast_spectra_proxy_coefficients.json",
    )
    p.add_argument(
        "--ai-spectra-root",
        default="/home/ecm5702/perm/ai_spectra",
        help="Root containing <expid>/spectra and <expid>/grb folders.",
    )
    p.add_argument("--expid", default="", help="If empty, pick first available expid from coefficients file.")
    p.add_argument("--param", default="2t")
    p.add_argument("--level", default="sfc")
    p.add_argument("--date", default="20230826")
    p.add_argument("--step", type=int, default=144)
    p.add_argument("--number", type=int, default=1)
    p.add_argument("--nside", type=int, default=128, help="Override nside if desired.")
    p.add_argument("--lmax", type=int, default=319, help="Override lmax if desired.")
    p.add_argument("--output-dir", default="/home/ecm5702/scratch/eval/spectra_proxy_validation/20260304_fast_proxy_overlay")
    p.add_argument("--auto-pick-sample", action="store_true", help="Pick first available sample for param/level.")
    return p.parse_args()


def _pick_sample(ai_spectra_root: Path, expids: list[str], param: str, level: str) -> dict[str, object]:
    dir_name = PARAM_TO_DIR[(param, level)]
    for expid in expids:
        spectra_dir = ai_spectra_root / expid / "spectra" / dir_name
        if not spectra_dir.exists():
            continue
        for ampl in sorted(spectra_dir.glob("ampl_*.npy")):
            m = AMPL_RE.match(ampl.name)
            if not m:
                continue
            if m.group("param") != param or m.group("level") != level or m.group("expid") != expid:
                continue
            return {
                "expid": expid,
                "date": m.group("date"),
                "step": int(m.group("step")),
                "number": int(m.group("number")),
            }
    raise FileNotFoundError(f"No spectra sample found for {param}_{level} under {ai_spectra_root}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = Path(args.coefficients_json)
    manifest = json.loads(coeff_path.read_text())
    coeffs = manifest.get("coefficients", {})
    group_key = f"{args.param}_{args.level}"
    if group_key not in coeffs:
        raise KeyError(f"Missing coefficients for {group_key} in {coeff_path}")

    beta = np.asarray(coeffs[group_key]["beta"], dtype=np.float64)
    model = str(coeffs[group_key]["model"])

    nside = args.nside if args.nside > 0 else int(manifest.get("nside", 128))
    lmax = args.lmax if args.lmax > 0 else int(manifest.get("lmax", 319))

    ai_spectra_root = Path(args.ai_spectra_root)
    expids = [str(x) for x in manifest.get("expids", [])]
    expid = args.expid
    date = args.date
    step = args.step
    number = args.number

    if args.auto_pick_sample:
        picked = _pick_sample(ai_spectra_root, expids, args.param, args.level)
        expid = str(picked["expid"])
        date = str(picked["date"])
        step = int(picked["step"])
        number = int(picked["number"])
    elif not expid:
        expid = expids[0] if expids else ""
        if not expid:
            raise ValueError("No expid provided and none found in coefficients file.")

    dir_name = PARAM_TO_DIR[(args.param, args.level)]
    ampl_path = ai_spectra_root / expid / "spectra" / dir_name / f"ampl_{date}_{step}_{args.param}_{args.level}_{expid}_n{number}.npy"
    wvn_path = ai_spectra_root / expid / "spectra" / dir_name / f"wvn_{date}_{step}_{args.param}_{args.level}_{expid}_n{number}.npy"
    grib_path = ai_spectra_root / expid / "grb" / dir_name / f"{expid}_{date}_{step}_{number}_nopoles.grb"
    if not ampl_path.exists() or not wvn_path.exists() or not grib_path.exists():
        raise FileNotFoundError(f"Missing sample files:\n  {ampl_path}\n  {wvn_path}\n  {grib_path}")

    ref_ampl = np.load(ampl_path)
    ref_wvn = np.load(wvn_path)
    lat, lon, values = read_grib_latlon_values(grib_path, args.param)
    cl = cl_from_unstructured(lat, lon, values, nside=nside, lmax=lmax)

    n = int(min(len(cl), len(ref_wvn), len(ref_ampl), lmax + 1))
    ell = np.arange(n, dtype=np.float64)
    proxy_ampl = apply_log_model(model, cl[:n], ell, beta)

    keep = (ell >= 2) & (ref_ampl[:n] > 0.0) & (proxy_ampl > 0.0) & np.isfinite(ref_ampl[:n]) & np.isfinite(proxy_ampl)
    if not np.any(keep):
        raise RuntimeError("No valid points available for overlay.")

    w = ref_wvn[:n][keep]
    a_ref = ref_ampl[:n][keep]
    a_proxy = proxy_ampl[keep]
    ratio = np.maximum(a_proxy, 1e-30) / np.maximum(a_ref, 1e-30)
    log10_abs = np.abs(np.log10(ratio))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(w, a_ref, color="#1f77b4", linewidth=2.0, label="Reference (metview)")
    ax.plot(w, a_proxy, color="#d62728", linewidth=1.8, linestyle="--", label=f"Proxy ({model})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Zonal wavenumber")
    ax.set_ylabel("Power amplitude")
    ax.set_title(f"{args.param} @ {args.level} | {expid} {date} step={step} n={number}")
    ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.6)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()

    stem = f"proxy_overlay_{expid}_{date}_{step}_{args.param}_{args.level}_n{number}"
    plot_path = out_dir / f"{stem}.pdf"
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "coefficients_json": str(coeff_path),
        "expid": expid,
        "param": args.param,
        "level": args.level,
        "date": date,
        "step": step,
        "number": number,
        "nside": nside,
        "lmax": lmax,
        "model": model,
        "mean_abs_log10_ratio": float(np.mean(log10_abs)),
        "median_abs_log10_ratio": float(np.median(log10_abs)),
        "p90_abs_log10_ratio": float(np.quantile(log10_abs, 0.9)),
        "median_ratio": float(np.median(ratio)),
        "plot_path": str(plot_path),
        "sample_paths": {
            "ampl_path": str(ampl_path),
            "wvn_path": str(wvn_path),
            "grib_path": str(grib_path),
        },
    }
    metrics_path = out_dir / f"{stem}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
