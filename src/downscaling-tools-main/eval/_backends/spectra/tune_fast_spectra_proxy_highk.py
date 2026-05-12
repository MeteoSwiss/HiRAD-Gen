from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from eval._backends.spectra.calibrate_fast_spectra_proxy import (
    aggregate_points,
    apply_log_model,
    collect_samples,
    fit_log_model,
)


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_design_matrix(model_name: str, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if model_name == "linear":
        return np.column_stack([np.ones_like(x1), x1, x2])
    if model_name == "poly2":
        return np.column_stack([np.ones_like(x1), x1, x2, x1 * x1, x2 * x2, x1 * x2])
    raise ValueError(f"Unknown model {model_name}")


def fit_log_model_weighted(model_name: str, cl: np.ndarray, ell: np.ndarray, ampl: np.ndarray, alpha: float, power: float) -> np.ndarray:
    eps = 1e-18
    x1 = np.log(np.maximum(cl, eps))
    x2 = np.log(np.maximum(ell + 1.0, 1.0))
    y = np.log(np.maximum(ampl, eps))

    X = _build_design_matrix(model_name, x1, x2)
    if np.max(ell) > 0:
        ell_norm = ell / np.max(ell)
    else:
        ell_norm = ell.copy()
    w = 1.0 + alpha * np.power(np.clip(ell_norm, 0.0, 1.0), power)
    sw = np.sqrt(np.maximum(w, 1e-12))

    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def eval_metrics(model_name: str, ell: np.ndarray, cl: np.ndarray, ampl_ref: np.ndarray, coeffs: np.ndarray, focus_ell_min: int) -> dict[str, float]:
    pred = apply_log_model(model_name, cl, ell, coeffs)
    ratio = np.maximum(pred, 1e-30) / np.maximum(ampl_ref, 1e-30)
    log10_abs = np.abs(np.log10(ratio))

    all_mask = ell >= 2
    high_mask = ell >= float(focus_ell_min)
    if not np.any(high_mask):
        high_mask = all_mask

    return {
        "all_mean_abs_log10_ratio": float(np.mean(log10_abs[all_mask])),
        "all_median_abs_log10_ratio": float(np.median(log10_abs[all_mask])),
        "all_p90_abs_log10_ratio": float(np.quantile(log10_abs[all_mask], 0.9)),
        "all_median_ratio": float(np.median(ratio[all_mask])),
        "high_mean_abs_log10_ratio": float(np.mean(log10_abs[high_mask])),
        "high_median_abs_log10_ratio": float(np.median(log10_abs[high_mask])),
        "high_p90_abs_log10_ratio": float(np.quantile(log10_abs[high_mask], 0.9)),
        "high_median_ratio": float(np.median(ratio[high_mask])),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune fast spectra proxy coefficients for high-k fidelity.")
    p.add_argument(
        "--base-coefficients-json",
        default="/home/ecm5702/scratch/eval/spectra_proxy_calibration/20260303_fast_proxy_bestof_v2_v3/fast_spectra_proxy_coefficients.json",
    )
    p.add_argument("--target-groups", default="t_850,z_500", help="Comma-separated param_level groups.")
    p.add_argument("--focus-ell-min", type=int, default=80)
    p.add_argument("--tradeoff-high", type=float, default=0.8, help="0..1 weight on high-k score when choosing best fit.")
    p.add_argument("--alphas", default="0,1,2,4,8,12,16")
    p.add_argument("--powers", default="1,2,3")
    p.add_argument("--models", default="poly2,linear")
    p.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/spectra_proxy_calibration/20260304_fast_proxy_highk_tz",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_coefficients_json)
    base_manifest = json.loads(base_path.read_text())
    base_coeffs = dict(base_manifest["coefficients"])

    expids = [str(x) for x in base_manifest.get("expids", [])]
    max_samples_per_param = int(base_manifest.get("max_samples_per_param", 12))
    seed = int(base_manifest.get("seed", 42))
    test_fraction = float(base_manifest.get("test_fraction", 0.25))
    nside = int(base_manifest.get("nside", 128))
    lmax = int(base_manifest.get("lmax", 319))

    target_groups = _parse_csv_strings(args.target_groups)
    alphas = _parse_csv_floats(args.alphas)
    powers = _parse_csv_floats(args.powers)
    models = _parse_csv_strings(args.models)

    samples = collect_samples(expids, max_samples_per_param=max_samples_per_param, seed=seed)
    if not samples:
        raise RuntimeError("No matching samples found for tuning.")

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = max(1, int(math.floor(len(shuffled) * test_fraction)))
    test_samples = shuffled[:n_test]
    train_samples = shuffled[n_test:] or test_samples

    train_points, _ = aggregate_points(train_samples, nside=nside, lmax=lmax)
    test_points, _ = aggregate_points(test_samples, nside=nside, lmax=lmax)

    tuned_coeffs = dict(base_coeffs)
    rows: list[dict[str, object]] = []
    picked: list[dict[str, object]] = []

    for key in target_groups:
        if key not in train_points:
            continue
        grp_train = train_points[key]
        grp_test = test_points.get(key)

        base_model = str(base_coeffs[key]["model"])
        base_beta = np.asarray(base_coeffs[key]["beta"], dtype=np.float64)
        base_train = eval_metrics(base_model, grp_train["ell"], grp_train["cl"], grp_train["ampl"], base_beta, args.focus_ell_min)
        base_test = (
            eval_metrics(base_model, grp_test["ell"], grp_test["cl"], grp_test["ampl"], base_beta, args.focus_ell_min)
            if grp_test is not None
            else None
        )
        base_score = (
            args.tradeoff_high * (base_test["high_mean_abs_log10_ratio"] if base_test else base_train["high_mean_abs_log10_ratio"])
            + (1.0 - args.tradeoff_high) * (base_test["all_mean_abs_log10_ratio"] if base_test else base_train["all_mean_abs_log10_ratio"])
        )

        best = {
            "score": base_score,
            "model": base_model,
            "alpha": 0.0,
            "power": 1.0,
            "beta": base_beta,
            "train": base_train,
            "test": base_test,
            "source": "baseline",
        }

        for model_name in models:
            for alpha in alphas:
                for power in powers:
                    if alpha == 0.0:
                        coeff = fit_log_model(model_name, grp_train["cl"], grp_train["ell"], grp_train["ampl"])
                    else:
                        coeff = fit_log_model_weighted(model_name, grp_train["cl"], grp_train["ell"], grp_train["ampl"], alpha, power)
                    tr = eval_metrics(model_name, grp_train["ell"], grp_train["cl"], grp_train["ampl"], coeff, args.focus_ell_min)
                    te = (
                        eval_metrics(model_name, grp_test["ell"], grp_test["cl"], grp_test["ampl"], coeff, args.focus_ell_min)
                        if grp_test is not None
                        else None
                    )
                    score = args.tradeoff_high * (te["high_mean_abs_log10_ratio"] if te else tr["high_mean_abs_log10_ratio"]) + (
                        1.0 - args.tradeoff_high
                    ) * (te["all_mean_abs_log10_ratio"] if te else tr["all_mean_abs_log10_ratio"])

                    rows.append(
                        {
                            "group": key,
                            "model": model_name,
                            "alpha": alpha,
                            "power": power,
                            "score": score,
                            "train_all_mean": tr["all_mean_abs_log10_ratio"],
                            "train_high_mean": tr["high_mean_abs_log10_ratio"],
                            "test_all_mean": te["all_mean_abs_log10_ratio"] if te else np.nan,
                            "test_high_mean": te["high_mean_abs_log10_ratio"] if te else np.nan,
                        }
                    )
                    if score < best["score"]:
                        best = {
                            "score": score,
                            "model": model_name,
                            "alpha": alpha,
                            "power": power,
                            "beta": coeff,
                            "train": tr,
                            "test": te,
                            "source": "tuned",
                        }

        tuned_coeffs[key] = {
            "model": str(best["model"]),
            "beta": [float(v) for v in np.asarray(best["beta"]).tolist()],
            "n_train_points": int(grp_train["ell"].size),
            "selected_from": str(best["source"]),
            "focus_ell_min": int(args.focus_ell_min),
            "tradeoff_high": float(args.tradeoff_high),
            "alpha": float(best["alpha"]),
            "power": float(best["power"]),
        }

        picked.append(
            {
                "group": key,
                "picked_source": best["source"],
                "picked_model": best["model"],
                "picked_alpha": best["alpha"],
                "picked_power": best["power"],
                "picked_score": best["score"],
                "baseline_score": base_score,
                "delta_score": best["score"] - base_score,
                "baseline_test_high_mean": base_test["high_mean_abs_log10_ratio"] if base_test else np.nan,
                "picked_test_high_mean": best["test"]["high_mean_abs_log10_ratio"] if best["test"] else np.nan,
                "baseline_test_all_mean": base_test["all_mean_abs_log10_ratio"] if base_test else np.nan,
                "picked_test_all_mean": best["test"]["all_mean_abs_log10_ratio"] if best["test"] else np.nan,
                "baseline_test_high_median_ratio": base_test["high_median_ratio"] if base_test else np.nan,
                "picked_test_high_median_ratio": best["test"]["high_median_ratio"] if best["test"] else np.nan,
            }
        )

    new_manifest = dict(base_manifest)
    new_manifest["source"] = "highk_tuned_from_bestof_v2_v3"
    new_manifest["focus_ell_min"] = int(args.focus_ell_min)
    new_manifest["tradeoff_high"] = float(args.tradeoff_high)
    new_manifest["alphas"] = alphas
    new_manifest["powers"] = powers
    new_manifest["coefficients"] = tuned_coeffs

    coeff_out = out_dir / "fast_spectra_proxy_coefficients.json"
    tuning_csv = out_dir / "tuning_grid_results.csv"
    picked_csv = out_dir / "picked_summary.csv"
    picked_md = out_dir / "picked_summary.md"

    coeff_out.write_text(json.dumps(new_manifest, indent=2))
    pd.DataFrame(rows).sort_values(["group", "score"]).to_csv(tuning_csv, index=False)
    picked_df = pd.DataFrame(picked).sort_values("group")
    picked_df.to_csv(picked_csv, index=False)

    md = []
    md.append("# High-k tuning summary")
    md.append("")
    md.append(f"- Base: `{base_path}`")
    md.append(f"- Focus ell min: `{args.focus_ell_min}`")
    md.append(f"- Tradeoff high: `{args.tradeoff_high}`")
    md.append("")
    md.append("| Group | Source | Model | alpha | power | baseline score | picked score | delta | baseline test high mean | picked test high mean |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in picked_df.iterrows():
        md.append(
            f"| {r['group']} | {r['picked_source']} | {r['picked_model']} | {r['picked_alpha']:.3f} | {r['picked_power']:.3f} | {r['baseline_score']:.6f} | {r['picked_score']:.6f} | {r['delta_score']:.6f} | {r['baseline_test_high_mean']:.6f} | {r['picked_test_high_mean']:.6f} |"
        )
    picked_md.write_text("\n".join(md) + "\n")

    print(f"Wrote tuned coefficients: {coeff_out}")
    print(f"Wrote tuning grid: {tuning_csv}")
    print(f"Wrote picked summary CSV: {picked_csv}")
    print(f"Wrote picked summary MD: {picked_md}")


if __name__ == "__main__":
    main()
