from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eval.paths import reference_spectra_dir
from eval._backends.spectra.calibrate_fast_spectra_proxy import (
    apply_log_model,
    cl_from_unstructured,
    read_grib_latlon_values,
)


PARAM_CONFIGS = [
    {"param": "2t", "level": "sfc", "dir_name": "2t_sfc"},
    {"param": "10u", "level": "sfc", "dir_name": "10u_sfc"},
    {"param": "10v", "level": "sfc", "dir_name": "10v_sfc"},
    {"param": "t", "level": "850", "dir_name": "t_850"},
    {"param": "z", "level": "500", "dir_name": "z_500"},
]
PARAM_BY_DIR = {c["dir_name"]: c for c in PARAM_CONFIGS}

AMPL_RE = re.compile(
    r"^ampl_(?P<date>\d{8})_(?P<step>\d+)_(?P<param>[a-z0-9]+)_(?P<level>[a-z0-9]+)_(?P<expid>[a-z0-9]+)_n(?P<number>\d+)\.npy$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ranking agreement between nice spectra and fast proxy spectra.")
    p.add_argument(
        "--expids",
        required=True,
        help="Comma-separated list of experiment IDs under /home/ecm5702/perm/ai_spectra.",
    )
    p.add_argument(
        "--fields",
        default="2t_sfc,10u_sfc,10v_sfc,t_850,z_500",
        help="Comma-separated field dir names.",
    )
    p.add_argument("--samples-per-field", type=int, default=5)
    p.add_argument(
        "--coefficients-json",
        default="/home/ecm5702/scratch/eval/spectra_proxy_calibration/20260304_fast_proxy_highk_tz/fast_spectra_proxy_coefficients.json",
    )
    p.add_argument("--ai-spectra-root", default="/home/ecm5702/perm/ai_spectra")
    p.add_argument("--reference-root", default=str(reference_spectra_dir("enfo_o320")))
    p.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/spectra_proxy_validation/20260304_proxy_ranking_agreement",
    )
    return p.parse_args()


def _spearman_rank_correlation(rank_a: list[int], rank_b: list[int]) -> float:
    n = len(rank_a)
    if n <= 1:
        return float("nan")
    diffsq = sum((a - b) ** 2 for a, b in zip(rank_a, rank_b))
    return 1.0 - (6.0 * diffsq) / (n * (n**2 - 1))


def _collect_tokens_for_exp(
    ai_spectra_root: Path,
    reference_root: Path,
    expid: str,
    field_dir: str,
    param: str,
    level: str,
) -> set[tuple[str, int, int]]:
    out: set[tuple[str, int, int]] = set()
    sdir = ai_spectra_root / expid / "spectra" / field_dir
    gdir = ai_spectra_root / expid / "grb" / field_dir
    if not sdir.exists() or not gdir.exists():
        return out

    for ampl_path in sorted(sdir.glob("ampl_*.npy")):
        m = AMPL_RE.match(ampl_path.name)
        if not m:
            continue
        if m.group("param") != param or m.group("level") != level or m.group("expid") != expid:
            continue
        date = m.group("date")
        step = int(m.group("step"))
        number = int(m.group("number"))
        wvn_path = sdir / f"wvn_{date}_{step}_{param}_{level}_{expid}_n{number}.npy"
        grib_path = gdir / f"{expid}_{date}_{step}_{number}_nopoles.grb"
        ref_ampl = reference_root / field_dir / f"ampl_{date}_{step}_{param}_{level}_1_n{number}.npy"
        if wvn_path.exists() and grib_path.exists() and ref_ampl.exists():
            out.add((date, step, number))
    return out


def _score_arrays(pred: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(pred), len(ref))
    if n < 4:
        return float("nan")
    p = np.asarray(pred[:n], dtype=np.float64)
    r = np.asarray(ref[:n], dtype=np.float64)
    ell = np.arange(n)
    keep = (ell >= 2) & np.isfinite(p) & np.isfinite(r) & (p > 0.0) & (r > 0.0)
    if not np.any(keep):
        return float("nan")
    ratio = np.maximum(p[keep], 1e-30) / np.maximum(r[keep], 1e-30)
    return float(np.mean(np.abs(np.log10(ratio))))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expids = [x.strip() for x in args.expids.split(",") if x.strip()]
    fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    ai_spectra_root = Path(args.ai_spectra_root)
    reference_root = Path(args.reference_root)
    coeff_manifest = json.loads(Path(args.coefficients_json).read_text())
    coeffs = coeff_manifest["coefficients"]

    sample_manifest: dict[str, list[dict[str, object]]] = {}
    score_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    cl_cache: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    nice_cache: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    ref_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
    proxy_cache: dict[tuple[str, str, str, int, int], np.ndarray] = {}

    for field_dir in fields:
        cfg = PARAM_BY_DIR[field_dir]
        param = cfg["param"]
        level = cfg["level"]
        group_key = f"{param}_{level}"
        if group_key not in coeffs:
            continue

        token_sets = [
            _collect_tokens_for_exp(ai_spectra_root, reference_root, expid, field_dir, param, level) for expid in expids
        ]
        common = set.intersection(*token_sets) if token_sets else set()
        chosen = sorted(common)[: args.samples_per_field]
        sample_manifest[field_dir] = [{"date": d, "step": s, "number": n} for (d, s, n) in chosen]

        if not chosen:
            continue

        model = str(coeffs[group_key]["model"])
        beta = np.asarray(coeffs[group_key]["beta"], dtype=np.float64)
        lmax = int(coeff_manifest.get("lmax", 319))
        nside = int(coeff_manifest.get("nside", 128))

        for expid in expids:
            nice_scores: list[float] = []
            proxy_scores: list[float] = []
            for date, step, number in chosen:
                nice_key = (expid, field_dir, date, step, number)
                ref_key = (field_dir, date, step, number)

                if nice_key not in nice_cache:
                    nice_path = ai_spectra_root / expid / "spectra" / field_dir / f"ampl_{date}_{step}_{param}_{level}_{expid}_n{number}.npy"
                    nice_cache[nice_key] = np.load(nice_path)
                if ref_key not in ref_cache:
                    ref_path = reference_root / field_dir / f"ampl_{date}_{step}_{param}_{level}_1_n{number}.npy"
                    ref_cache[ref_key] = np.load(ref_path)

                nice_ampl = nice_cache[nice_key]
                ref_ampl = ref_cache[ref_key]
                nice_score = _score_arrays(nice_ampl, ref_ampl)
                if math.isfinite(nice_score):
                    nice_scores.append(nice_score)

                proxy_key = (expid, field_dir, date, step, number)
                if proxy_key not in proxy_cache:
                    cl_key = (expid, field_dir, date, step, number)
                    if cl_key not in cl_cache:
                        grib_path = ai_spectra_root / expid / "grb" / field_dir / f"{expid}_{date}_{step}_{number}_nopoles.grb"
                        lat, lon, values = read_grib_latlon_values(grib_path, param)
                        cl_cache[cl_key] = cl_from_unstructured(lat, lon, values, nside=nside, lmax=lmax)
                    cl = cl_cache[cl_key]
                    n = min(len(cl), len(ref_ampl), lmax + 1)
                    ell = np.arange(n, dtype=np.float64)
                    proxy_cache[proxy_key] = apply_log_model(model, cl[:n], ell, beta)

                proxy_score = _score_arrays(proxy_cache[proxy_key], ref_ampl)
                if math.isfinite(proxy_score):
                    proxy_scores.append(proxy_score)

            row = {
                "field": field_dir,
                "expid": expid,
                "samples_used": len(chosen),
                "nice_score_mean_abs_log10_ratio": float(np.mean(nice_scores)) if nice_scores else np.nan,
                "proxy_score_mean_abs_log10_ratio": float(np.mean(proxy_scores)) if proxy_scores else np.nan,
            }
            score_rows.append(row)

        field_df = pd.DataFrame([r for r in score_rows if r["field"] == field_dir]).copy()
        field_df = field_df.dropna(subset=["nice_score_mean_abs_log10_ratio", "proxy_score_mean_abs_log10_ratio"])
        if field_df.empty:
            continue
        field_df = field_df.sort_values("nice_score_mean_abs_log10_ratio").reset_index(drop=True)
        field_df["nice_rank"] = np.arange(1, len(field_df) + 1)
        proxy_rank_map = (
            field_df.sort_values("proxy_score_mean_abs_log10_ratio")
            .reset_index(drop=True)
            .assign(proxy_rank=lambda x: np.arange(1, len(x) + 1))
            .set_index("expid")["proxy_rank"]
            .to_dict()
        )
        field_df["proxy_rank"] = field_df["expid"].map(proxy_rank_map).astype(int)

        rho = _spearman_rank_correlation(field_df["nice_rank"].tolist(), field_df["proxy_rank"].tolist())
        exact_match = list(field_df.sort_values("nice_rank")["expid"]) == list(
            field_df.sort_values("proxy_rank")["expid"]
        )
        top1_nice = str(field_df.sort_values("nice_rank").iloc[0]["expid"])
        top1_proxy = str(field_df.sort_values("proxy_rank").iloc[0]["expid"])
        summary_rows.append(
            {
                "field": field_dir,
                "n_experiments": int(len(field_df)),
                "samples_per_field": int(len(chosen)),
                "spearman_rho": float(rho),
                "exact_order_match": bool(exact_match),
                "top1_match": bool(top1_nice == top1_proxy),
                "top1_nice": top1_nice,
                "top1_proxy": top1_proxy,
            }
        )

    scores_df = pd.DataFrame(score_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("field")
    samples_path = output_dir / "sample_manifest.json"
    scores_path = output_dir / "scores_by_experiment.csv"
    summary_path = output_dir / "ranking_agreement_summary.csv"
    report_path = output_dir / "ranking_agreement_report.md"

    samples_path.write_text(json.dumps(sample_manifest, indent=2))
    scores_df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    md: list[str] = []
    md.append("# Proxy Ranking Agreement Report")
    md.append("")
    md.append(f"- Coefficients: `{args.coefficients_json}`")
    md.append(f"- Reference: `{reference_root}`")
    md.append(f"- Experiments ({len(expids)}): `{','.join(expids)}`")
    md.append(f"- Fields: `{','.join(fields)}`")
    md.append(f"- Requested samples/field: `{args.samples_per_field}`")
    md.append("")
    md.append("## Field-Level Agreement")
    md.append("")
    md.append("| Field | N exp | Samples | Spearman rho | Exact order match | Top1 nice | Top1 proxy | Top1 match |")
    md.append("|---|---:|---:|---:|---|---|---|---|")
    for _, r in summary_df.iterrows():
        md.append(
            f"| {r['field']} | {int(r['n_experiments'])} | {int(r['samples_per_field'])} | {float(r['spearman_rho']):.3f} | {bool(r['exact_order_match'])} | {r['top1_nice']} | {r['top1_proxy']} | {bool(r['top1_match'])} |"
        )
    md.append("")

    for field_dir in fields:
        fdf = scores_df[scores_df["field"] == field_dir].dropna(
            subset=["nice_score_mean_abs_log10_ratio", "proxy_score_mean_abs_log10_ratio"]
        )
        if fdf.empty:
            continue
        fdf = fdf.copy()
        nice_order = (
            fdf.sort_values("nice_score_mean_abs_log10_ratio").reset_index(drop=True).assign(nice_rank=lambda x: np.arange(1, len(x) + 1))
        )
        proxy_order = (
            fdf.sort_values("proxy_score_mean_abs_log10_ratio")
            .reset_index(drop=True)
            .assign(proxy_rank=lambda x: np.arange(1, len(x) + 1))[["expid", "proxy_rank"]]
        )
        merged = nice_order.merge(proxy_order, on="expid", how="left")
        md.append(f"## {field_dir} Ranking")
        md.append("")
        md.append("| Nice rank | Proxy rank | Experiment | Nice score | Proxy score |")
        md.append("|---:|---:|---|---:|---:|")
        for _, r in merged.iterrows():
            md.append(
                f"| {int(r['nice_rank'])} | {int(r['proxy_rank'])} | {r['expid']} | {float(r['nice_score_mean_abs_log10_ratio']):.6f} | {float(r['proxy_score_mean_abs_log10_ratio']):.6f} |"
            )
        md.append("")

    report_path.write_text("\n".join(md) + "\n")

    print(f"Wrote sample manifest: {samples_path}")
    print(f"Wrote scores: {scores_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
