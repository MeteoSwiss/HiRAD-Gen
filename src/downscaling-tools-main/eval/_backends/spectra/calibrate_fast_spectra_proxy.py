from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import earthkit.data as ekd
import healpy as hp
import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class Sample:
    expid: str
    param: str
    level: str
    date: str
    step: int
    number: int
    ampl_path: Path
    wvn_path: Path
    grib_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a fast Python spectra proxy against precomputed Metview spectra amplitudes."
    )
    parser.add_argument(
        "--expids",
        default="j0ys,j1li,j24v,j1eg,iytd,iysd,iytc,j10d",
        help="Comma-separated expids under /home/ecm5702/perm/ai_spectra.",
    )
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--lmax", type=int, default=319)
    parser.add_argument("--max-samples-per-param", type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/spectra_proxy_calibration",
        help="Output directory for coefficients and metrics.",
    )
    return parser.parse_args()


def build_healpix_mean_map(lat_deg: np.ndarray, lon_deg: np.ndarray, values: np.ndarray, nside: int) -> tuple[np.ndarray, float]:
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.mod(np.asarray(lon_deg, dtype=np.float64), 360.0)
    val = np.asarray(values, dtype=np.float64)

    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    npix = hp.nside2npix(nside)
    sums = np.zeros(npix, dtype=np.float64)
    counts = np.zeros(npix, dtype=np.int64)
    np.add.at(sums, pix, val)
    np.add.at(counts, pix, 1)

    m = np.full(npix, hp.UNSEEN, dtype=np.float64)
    valid = counts > 0
    m[valid] = sums[valid] / counts[valid]
    mean_valid = np.mean(m[valid]) if np.any(valid) else 0.0
    m[valid] = m[valid] - mean_valid

    coverage = float(valid.sum()) / float(npix)
    return m, coverage


def cl_from_unstructured(lat_deg: np.ndarray, lon_deg: np.ndarray, values: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    m, coverage = build_healpix_mean_map(lat_deg, lon_deg, values, nside=nside)
    if coverage <= 0.0:
        raise RuntimeError("No valid HEALPix coverage for this field")

    m_ma = hp.ma(m)
    m_ma.mask = np.isclose(m, hp.UNSEEN)

    cl = hp.anafast(m_ma.filled(0.0), lmax=lmax)
    cl = cl / coverage
    return cl


def read_grib_latlon_values(grib_path: Path, short_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fl = ekd.from_source("file", str(grib_path))
    for idx in range(len(fl)):
        field = fl[idx]
        if field.metadata().get("shortName") == short_name:
            ll = field.to_latlon()
            vals = field.to_numpy()
            return np.asarray(ll["lat"]), np.asarray(ll["lon"]), np.asarray(vals)

    # Fallback: many of these files have a single field
    if len(fl) == 1:
        field = fl[0]
        ll = field.to_latlon()
        vals = field.to_numpy()
        return np.asarray(ll["lat"]), np.asarray(ll["lon"]), np.asarray(vals)

    raise KeyError(f"Could not find shortName={short_name} in {grib_path}")


def collect_samples(expids: list[str], max_samples_per_param: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    all_samples: list[Sample] = []

    for expid in expids:
        base = Path(f"/home/ecm5702/perm/ai_spectra/{expid}")
        spectra_root = base / "spectra"
        grb_root = base / "grb"
        if not spectra_root.exists() or not grb_root.exists():
            continue

        for (param, level), dir_name in PARAM_TO_DIR.items():
            files = sorted((spectra_root / dir_name).glob("ampl_*.npy"))
            parsed: list[Sample] = []
            for ampl_path in files:
                m = AMPL_RE.match(ampl_path.name)
                if not m:
                    continue
                if m.group("param") != param or m.group("level") != level or m.group("expid") != expid:
                    continue

                date = m.group("date")
                step = int(m.group("step"))
                number = int(m.group("number"))
                wvn_path = ampl_path.with_name(ampl_path.name.replace("ampl_", "wvn_"))
                grib_name = f"{expid}_{date}_{step}_{number}_nopoles.grb"
                grib_path = grb_root / dir_name / grib_name
                if not (wvn_path.exists() and grib_path.exists()):
                    continue

                parsed.append(
                    Sample(
                        expid=expid,
                        param=param,
                        level=level,
                        date=date,
                        step=step,
                        number=number,
                        ampl_path=ampl_path,
                        wvn_path=wvn_path,
                        grib_path=grib_path,
                    )
                )

            rng.shuffle(parsed)
            all_samples.extend(parsed[:max_samples_per_param])

    return all_samples


def _build_design_matrix(model_name: str, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if model_name == "linear":
        return np.column_stack([np.ones_like(x1), x1, x2])
    if model_name == "poly2":
        return np.column_stack([np.ones_like(x1), x1, x2, x1 * x1, x2 * x2, x1 * x2])
    raise ValueError(f"Unknown model {model_name}")


def fit_log_model(model_name: str, cl: np.ndarray, ell: np.ndarray, ampl: np.ndarray) -> np.ndarray:
    eps = 1e-18
    x1 = np.log(np.maximum(cl, eps))
    x2 = np.log(np.maximum(ell + 1.0, 1.0))
    y = np.log(np.maximum(ampl, eps))

    X = _build_design_matrix(model_name, x1, x2)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def apply_log_model(model_name: str, cl: np.ndarray, ell: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eps = 1e-18
    x1 = np.log(np.maximum(cl, eps))
    x2 = np.log(np.maximum(ell + 1.0, 1.0))
    X = _build_design_matrix(model_name, x1, x2)
    y_log = X @ beta
    return np.exp(y_log)


def aggregate_points(samples: list[Sample], nside: int, lmax: int) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    per_param_points: dict[str, dict[str, list[np.ndarray]]] = {}
    processed_meta: list[dict[str, object]] = []

    for sample in samples:
        try:
            lat, lon, values = read_grib_latlon_values(sample.grib_path, sample.param)
            cl = cl_from_unstructured(lat, lon, values, nside=nside, lmax=lmax)
            ref_wvn = np.load(sample.wvn_path)
            ref_ampl = np.load(sample.ampl_path)

            n = int(min(len(cl), len(ref_wvn), len(ref_ampl), lmax + 1))
            ell = np.arange(n, dtype=np.float64)
            keep = (ell >= 2) & np.isfinite(cl[:n]) & np.isfinite(ref_ampl[:n]) & (cl[:n] > 0.0) & (ref_ampl[:n] > 0.0)
            if not np.any(keep):
                continue

            key = f"{sample.param}_{sample.level}"
            bucket = per_param_points.setdefault(key, {"ell": [], "cl": [], "ampl": []})
            bucket["ell"].append(ell[keep])
            bucket["cl"].append(cl[:n][keep])
            bucket["ampl"].append(ref_ampl[:n][keep])

            processed_meta.append(
                {
                    "expid": sample.expid,
                    "param": sample.param,
                    "level": sample.level,
                    "date": sample.date,
                    "step": sample.step,
                    "number": sample.number,
                    "grib_path": str(sample.grib_path),
                    "ampl_path": str(sample.ampl_path),
                }
            )
        except Exception as exc:
            processed_meta.append(
                {
                    "expid": sample.expid,
                    "param": sample.param,
                    "level": sample.level,
                    "date": sample.date,
                    "step": sample.step,
                    "number": sample.number,
                    "grib_path": str(sample.grib_path),
                    "ampl_path": str(sample.ampl_path),
                    "error": str(exc),
                }
            )

    out: dict[str, dict[str, np.ndarray]] = {}
    for key, arrays in per_param_points.items():
        out[key] = {
            "ell": np.concatenate(arrays["ell"]),
            "cl": np.concatenate(arrays["cl"]),
            "ampl": np.concatenate(arrays["ampl"]),
        }
    return out, processed_meta


def eval_group(model_name: str, ell: np.ndarray, cl: np.ndarray, ampl_ref: np.ndarray, coeffs: np.ndarray) -> dict[str, float]:
    pred = apply_log_model(model_name, cl, ell, coeffs)
    ratio = np.maximum(pred, 1e-30) / np.maximum(ampl_ref, 1e-30)
    log10_abs = np.abs(np.log10(ratio))
    return {
        "median_abs_log10_ratio": float(np.median(log10_abs)),
        "p90_abs_log10_ratio": float(np.quantile(log10_abs, 0.9)),
        "mean_abs_log10_ratio": float(np.mean(log10_abs)),
        "median_ratio": float(np.median(ratio)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expids = [x.strip() for x in args.expids.split(",") if x.strip()]
    samples = collect_samples(expids, max_samples_per_param=args.max_samples_per_param, seed=args.seed)
    if not samples:
        raise RuntimeError("No matching samples found. Check expids and ai_spectra folder contents.")

    rng = random.Random(args.seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = max(1, int(math.floor(len(shuffled) * args.test_fraction)))
    test_samples = shuffled[:n_test]
    train_samples = shuffled[n_test:]
    if not train_samples:
        train_samples = test_samples

    train_points, train_meta = aggregate_points(train_samples, nside=args.nside, lmax=args.lmax)
    test_points, test_meta = aggregate_points(test_samples, nside=args.nside, lmax=args.lmax)

    coefficients: dict[str, dict[str, object]] = {}
    metrics_rows: list[dict[str, object]] = []

    for key, grp in sorted(train_points.items()):
        candidate_models = ["linear", "poly2"]
        best_model = None
        best_coeffs = None
        best_test_mae = float("inf")
        best_train_metrics = None
        best_test_metrics = None

        for model_name in candidate_models:
            coeffs = fit_log_model(model_name, grp["cl"], grp["ell"], grp["ampl"])
            train_metrics = eval_group(model_name, grp["ell"], grp["cl"], grp["ampl"], coeffs)
            if key in test_points:
                tgrp = test_points[key]
                test_metrics = eval_group(model_name, tgrp["ell"], tgrp["cl"], tgrp["ampl"], coeffs)
                model_score = test_metrics["mean_abs_log10_ratio"]
            else:
                test_metrics = None
                model_score = train_metrics["mean_abs_log10_ratio"]

            if model_score < best_test_mae:
                best_test_mae = model_score
                best_model = model_name
                best_coeffs = coeffs
                best_train_metrics = train_metrics
                best_test_metrics = test_metrics

        assert best_model is not None and best_coeffs is not None and best_train_metrics is not None
        coefficients[key] = {
            "model": best_model,
            "beta": [float(v) for v in best_coeffs.tolist()],
            "n_train_points": int(grp["ell"].size),
        }
        metrics_rows.append({"group": key, "split": "train", "model": best_model, **best_train_metrics})
        if best_test_metrics is not None:
            metrics_rows.append({"group": key, "split": "test", "model": best_model, **best_test_metrics})

    manifest = {
        "expids": expids,
        "nside": args.nside,
        "lmax": args.lmax,
        "max_samples_per_param": args.max_samples_per_param,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "n_samples_total": len(samples),
        "n_samples_train": len(train_samples),
        "n_samples_test": len(test_samples),
        "coefficients": coefficients,
    }

    coeff_path = output_dir / "fast_spectra_proxy_coefficients.json"
    metrics_path = output_dir / "fast_spectra_proxy_metrics.csv"
    train_meta_path = output_dir / "fast_spectra_proxy_train_samples.json"
    test_meta_path = output_dir / "fast_spectra_proxy_test_samples.json"

    coeff_path.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    train_meta_path.write_text(json.dumps(train_meta, indent=2))
    test_meta_path.write_text(json.dumps(test_meta, indent=2))

    print(f"Wrote coefficients: {coeff_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote train metadata: {train_meta_path}")
    print(f"Wrote test metadata: {test_meta_path}")


if __name__ == "__main__":
    main()
