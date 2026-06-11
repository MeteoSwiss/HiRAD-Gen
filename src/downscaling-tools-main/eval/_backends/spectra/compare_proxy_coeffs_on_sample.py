from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval._backends.spectra.calibrate_fast_spectra_proxy import (
    apply_log_model,
    cl_from_unstructured,
    read_grib_latlon_values,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two proxy coefficient packages on one sample.")
    p.add_argument("--baseline-coefficients-json", required=True)
    p.add_argument("--candidate-coefficients-json", required=True)
    p.add_argument("--expid", default="j0ys")
    p.add_argument("--date", default="20230801")
    p.add_argument("--step", type=int, default=144)
    p.add_argument("--number", type=int, default=1)
    p.add_argument("--groups", default="t_850,z_500", help="Comma-separated param_level.")
    p.add_argument("--ai-spectra-root", default="/home/ecm5702/perm/ai_spectra")
    p.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/spectra_proxy_validation/20260304_fast_proxy_overlay_j0ys_highk_tuned",
    )
    p.add_argument("--focus-k-min", type=int, default=40)
    return p.parse_args()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _evaluate_group(
    manifest: dict,
    key: str,
    cl: np.ndarray,
    ell: np.ndarray,
    ref_ampl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    info = manifest["coefficients"][key]
    model = str(info["model"])
    beta = np.asarray(info["beta"], dtype=np.float64)
    pred = apply_log_model(model, cl, ell, beta)
    ratio = np.maximum(pred, 1e-30) / np.maximum(ref_ampl, 1e-30)
    err = np.abs(np.log10(ratio))
    return ratio, err


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline_coefficients_json)
    candidate_path = Path(args.candidate_coefficients_json)
    baseline = _load_manifest(baseline_path)
    candidate = _load_manifest(candidate_path)

    ai_root = Path(args.ai_spectra_root) / args.expid
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    bands = [("all", 2), ("k>=40", 40), ("k>=80", 80), ("k>=120", 120)]
    band_order = {"all": 0, "k>=40": 1, "k>=80": 2, "k>=120": 3}

    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(groups), 1, figsize=(8.4, 3.2 * max(1, len(groups))), sharex=True)
    if len(groups) == 1:
        axes = [axes]

    for i, group in enumerate(groups):
        param, level = group.split("_", 1)
        ampl_path = ai_root / "spectra" / group / f"ampl_{args.date}_{args.step}_{param}_{level}_{args.expid}_n{args.number}.npy"
        wvn_path = ai_root / "spectra" / group / f"wvn_{args.date}_{args.step}_{param}_{level}_{args.expid}_n{args.number}.npy"
        grib_path = ai_root / "grb" / group / f"{args.expid}_{args.date}_{args.step}_{args.number}_nopoles.grb"

        ref_ampl = np.load(ampl_path)
        wvn = np.load(wvn_path)
        lat, lon, val = read_grib_latlon_values(grib_path, param)
        cl = cl_from_unstructured(lat, lon, val, nside=128, lmax=319)

        m = min(len(cl), len(ref_ampl), len(wvn), 320)
        ell = np.arange(m, dtype=np.float64)
        ref = ref_ampl[:m]
        ww = wvn[:m]
        clm = cl[:m]

        ratio_b, err_b = _evaluate_group(baseline, group, clm, ell, ref)
        ratio_c, err_c = _evaluate_group(candidate, group, clm, ell, ref)

        keep = (ell >= 2) & (ref > 0) & np.isfinite(ref) & np.isfinite(ratio_b) & np.isfinite(ratio_c) & (ratio_b > 0) & (ratio_c > 0)
        ee = ell[keep]
        ww = ww[keep]
        ratio_b = ratio_b[keep]
        ratio_c = ratio_c[keep]
        err_b = err_b[keep]
        err_c = err_c[keep]

        for band_name, kmin in bands:
            mask = ee >= float(kmin)
            if not np.any(mask):
                continue
            rows.append(
                {
                    "group": group,
                    "band": band_name,
                    "baseline_mean": float(np.mean(err_b[mask])),
                    "candidate_mean": float(np.mean(err_c[mask])),
                    "delta_mean": float(np.mean(err_c[mask]) - np.mean(err_b[mask])),
                    "baseline_p90": float(np.quantile(err_b[mask], 0.9)),
                    "candidate_p90": float(np.quantile(err_c[mask], 0.9)),
                    "delta_p90": float(np.quantile(err_c[mask], 0.9) - np.quantile(err_b[mask], 0.9)),
                    "baseline_median_ratio": float(np.median(ratio_b[mask])),
                    "candidate_median_ratio": float(np.median(ratio_c[mask])),
                }
            )

        ax = axes[i]
        focus = ee >= float(args.focus_k_min)
        ax.plot(ww[focus], ratio_b[focus], color="#d62728", lw=1.5, alpha=0.85, label="baseline")
        ax.plot(ww[focus], ratio_c[focus], color="#2ca02c", lw=1.5, alpha=0.90, label="candidate")
        ax.axhline(1.0, color="k", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0.75, 1.35)
        ax.set_ylabel(f"{group}\nproxy/reference")
        ax.grid(ls="--", lw=0.3, alpha=0.6)
        ax.legend(frameon=False, fontsize=8, loc="best")

    axes[-1].set_xlabel("zonal wavenumber")
    fig.suptitle(f"{args.expid} ratio comparison (k>={args.focus_k_min}) baseline vs candidate", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    ratio_pdf = output_dir / f"{args.expid}_ratio_baseline_vs_candidate.pdf"
    ratio_png = output_dir / f"{args.expid}_ratio_baseline_vs_candidate.png"
    fig.savefig(ratio_pdf, dpi=230, bbox_inches="tight")
    fig.savefig(ratio_png, dpi=230, bbox_inches="tight")
    plt.close(fig)

    rows = sorted(rows, key=lambda r: (str(r["group"]), band_order[str(r["band"])]))
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{args.expid}_baseline_vs_candidate_metrics.csv"
    df.to_csv(csv_path, index=False)

    md = []
    md.append(f"# {args.expid} high-k coefficient comparison")
    md.append("")
    md.append(f"- Baseline: `{baseline_path}`")
    md.append(f"- Candidate: `{candidate_path}`")
    md.append(f"- Sample: date={args.date} step={args.step} n={args.number}")
    md.append("")
    md.append("| Group | Band | Baseline mean | Candidate mean | Delta mean | Baseline p90 | Candidate p90 | Delta p90 | Baseline median ratio | Candidate median ratio |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['group']} | {r['band']} | {r['baseline_mean']:.6f} | {r['candidate_mean']:.6f} | {r['delta_mean']:.6f} | {r['baseline_p90']:.6f} | {r['candidate_p90']:.6f} | {r['delta_p90']:.6f} | {r['baseline_median_ratio']:.6f} | {r['candidate_median_ratio']:.6f} |"
        )
    md.append("")
    md.append("## Diagnostics")
    md.append(f"- `{ratio_pdf}`")
    md.append(f"- `{ratio_png}`")
    md_path = output_dir / f"{args.expid}_highk_comparison.md"
    md_path.write_text("\n".join(md) + "\n")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote MD: {md_path}")
    print(f"Wrote ratio PDF: {ratio_pdf}")
    print(f"Wrote ratio PNG: {ratio_png}")


if __name__ == "__main__":
    import pandas as pd

    main()
