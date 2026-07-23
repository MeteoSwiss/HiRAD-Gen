"""
Overlay multiple models on a single set of diurnal-cycle and probability-of-
exceedance figures, and emit a scalar scoreboard quantifying how close each model
is to the target ("did we beat CorrDiff?").

Unlike the per-model eval scripts (diurnal_cycle_*.py, probability_of_exceedance.py)
which read ONE `inference_output_dir` and hardcode the legend label "CorrDiff",
this script takes a LIST of models, restricts to the timestamps present in ALL of
them (apples-to-apples), and overlays them with the target.

It streams one timestep at a time (per model) so it does not OOM on long windows
— the all-in-memory approach in diurnal_cycle_precip_p99.py does not scale.

Metrics computed per model vs target:
  * diurnal_mean_rmse      – RMSE of the 24-h domain-mean precip diurnal cycle
  * diurnal_p99_rmse       – RMSE of the 24-h P99 precip diurnal cycle
  * exceedance_log_err     – mean |log10(P_exc_model) - log10(P_exc_target)| over
                             thresholds where the target exceedance is resolvable
Lower is better for all three; the winner per metric is flagged.

Also reported (absolute, NOT scored vs target — the target is deterministic):
  * ensemble_spread        – mean per-pixel std across ensemble members (mm/h),
                             averaged over land and time. The dispersion diagnostic:
                             compare across a sigma_max sweep, or DiT vs CorrDiff,
                             to see whether over-dispersion is sampler-injected.

Config (YAML, passed via --config-name) — see conf/compare_models.yaml:
  models:        list of {name, dir}
  target_from:   model name whose `-target` files define the reference (default: first)
  times_range:   optional [start, end, step]; otherwise the common set is used
  conv_factor / conv_factor_hourly / land_sea_mask_path / height / width / log_interval
  output_dir:    where overlay PNGs + scoreboard.json are written
  cache:         (default true) reuse per-model summaries from output_dir/summary_cache/
                 when the timestep set, factors, mask and tensor mtimes are unchanged —
                 a rerun with one new model only streams that model

Usage
-----
python -m hirad.eval.compare_models --config-name=src/hirad/conf/compare_models.yaml
# quick override of the model list (name=dir pairs):
python -m hirad.eval.compare_models --config-name=... \
    --models corrdiff=/path/cd dit_edm=/path/dit zeros=/path/ar
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hirad.datasets import known_datasets
from hirad.utils.function_utils import get_time_from_range
from hirad.eval.plotting import get_channel_indices, load_land_sea_mask

logger = logging.getLogger("compare_models")

# Exceedance thresholds (mm/h), matching probability_of_exceedance.py
THRESHOLDS = np.logspace(-2, 3.0, 200)
# Floor for the target exceedance probability below which it is too noisy to score.
EXC_SCORE_FLOOR = 1e-6


# Bump when the summary computation changes (new metrics, changed accumulators) —
# invalidates all existing caches.
CACHE_VERSION = 2


def _cache_signature(model_dir, times, conv_factor, conv_factor_hourly,
                     is_target_only, n_land) -> str:
    """Fingerprint of everything a cached per-model summary depends on.

    Includes the exact timestep set (metrics are computed over the COMMON
    intersection — a different set invalidates), the conversion factors, the
    land-pixel count, the threshold grid, and the mtimes of a sample of the
    model's tensor files (so regenerating a dir in place invalidates too).
    """
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|{len(THRESHOLDS)}|{THRESHOLDS[0]}|{THRESHOLDS[-1]}".encode())
    h.update(f"|{conv_factor}|{conv_factor_hourly}|{int(is_target_only)}|{n_land}".encode())
    h.update("|".join(times).encode())
    suffix = "-target" if is_target_only else "-predictions"
    root = Path(model_dir)
    for ts in (times[0], times[len(times) // 2], times[-1]):
        f = root / ts / f"{ts}{suffix}"
        h.update(str(f.stat().st_mtime_ns if f.exists() else -1).encode())
    return h.hexdigest()[:16]


def _cache_load(cache_dir: Path, name: str, sig: str):
    path = cache_dir / f"{name}.{sig}.json"
    if not path.is_file():
        return None
    with open(path) as f:
        raw = json.load(f)
    out = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in raw.items()}
    logger.info(f"[{name}] loaded cached summaries ({path.name})")
    return out


def _cache_save(cache_dir: Path, name: str, sig: str, result: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items()}
    path = cache_dir / f"{name}.{sig}.json"
    with open(path, "w") as f:
        json.dump(serializable, f)
    logger.info(f"[{name}] cached summaries -> {path.name}")


def _available_timestamps(model_dir: Path, kind: str = "predictions") -> set[str]:
    """Timestamps in `model_dir` that have the requested file.

    kind="predictions" (default) → a ``{ts}-predictions`` file, which is all a
    prediction model needs. kind="target" → a ``{ts}-target`` file, required only
    from the target source: the ground-truth target is identical across models and
    is read once, so it is NOT duplicated into every model's output dir.
    """
    out = set()
    if not model_dir.is_dir():
        return out
    for sub in model_dir.iterdir():
        if not sub.is_dir():
            continue
        ts = sub.name
        if (sub / f"{ts}-{kind}").exists():
            out.add(ts)
    return out


def _hourly_finalize(sum_v, sum_std, cnt):
    """Return (mean[24], std[24]) from per-hour accumulators, NaN where cnt==0."""
    mean = np.full(24, np.nan)
    std = np.full(24, np.nan)
    nz = cnt > 0
    mean[nz] = sum_v[nz] / cnt[nz]
    std[nz] = sum_std[nz] / cnt[nz]
    return mean, std


def _cyclic(arr):
    """Append hour-0 value at hour-24 for a closed diurnal curve."""
    return np.append(arr, arr[0])


def process_model(name, model_dir, times, tp_out, conv_factor, conv_factor_hourly,
                  land_bool, log_interval, is_target_only=False):
    """Single streaming pass over `times`; returns diurnal + exceedance summaries.

    For `is_target_only` we read the `-target` file (single field). Otherwise we
    read the `-predictions` ensemble and compute member statistics.
    """
    out_root = Path(model_dir)
    n_thr = len(THRESHOLDS)

    # Diurnal accumulators (per hour): sum of per-ts (mean-over-members) and of
    # per-ts (std-over-members), matching the existing eval semantics exactly.
    sum_mean, sum_mean_std, cnt = (np.zeros(24) for _ in range(3))
    sum_p99, sum_p99_std = np.zeros(24), np.zeros(24)
    # Exceedance accumulators (ensemble-mean over members).
    exc_counts = np.zeros(n_thr, dtype=np.float64)
    total = 0
    # Per-pixel ensemble spread accumulator (mm/h); ensembles only.
    sum_spread, n_spread = 0.0, 0

    for i, ts in enumerate(times):
        hour = datetime.strptime(ts, "%Y%m%d-%H%M").hour

        if is_target_only:
            tgt = torch.load(out_root / ts / f"{ts}-target", weights_only=False)
            field_day = np.asarray(tgt[tp_out]) * conv_factor          # mm/day (H,W)
            field_hr = np.asarray(tgt[tp_out]) * conv_factor_hourly    # mm/h  (H,W)
            vals_day = field_day[land_bool][None, :]                   # (1, n_land)
            vals_hr = field_hr[land_bool][None, :]
        else:
            preds = torch.load(out_root / ts / f"{ts}-predictions", weights_only=False)
            preds = np.asarray(preds)                                  # (M, C, H, W)
            vals_day = preds[:, tp_out][:, land_bool] * conv_factor    # (M, n_land)
            vals_hr = preds[:, tp_out][:, land_bool] * conv_factor_hourly

        # Diurnal mean (mm/day): mean over land per member, then over members.
        member_means = vals_day.mean(axis=1)                          # (M,)
        member_p99 = np.quantile(vals_day, 0.99, axis=1)              # (M,)
        sum_mean[hour] += member_means.mean()
        sum_mean_std[hour] += member_means.std()
        sum_p99[hour] += member_p99.mean()
        sum_p99_std[hour] += member_p99.std()
        cnt[hour] += 1

        # Exceedance (mm/h): per-member counts, averaged over members.
        m = vals_hr.shape[0]
        per_member_exc = np.zeros(n_thr, dtype=np.float64)
        for mi in range(m):
            per_member_exc += np.sum(vals_hr[mi][:, None] > THRESHOLDS[None, :], axis=0)
        exc_counts += per_member_exc / m
        total += vals_hr.shape[1]

        # Per-pixel ensemble spread (mm/h): std across members at each land pixel,
        # averaged over land. Only meaningful for ensembles (target is one field).
        if not is_target_only and m > 1:
            sum_spread += float(vals_hr.std(axis=0).mean())
            n_spread += 1

        if log_interval and (i % log_interval == 0 or i == len(times) - 1):
            logger.info(f"[{name}] {i + 1}/{len(times)} ({ts})")

    mean, mean_std = _hourly_finalize(sum_mean, sum_mean_std, cnt)
    p99, p99_std = _hourly_finalize(sum_p99, sum_p99_std, cnt)
    exceedance = exc_counts / total if total > 0 else np.full(n_thr, np.nan)
    ensemble_spread = sum_spread / n_spread if n_spread > 0 else float("nan")

    return {
        "diurnal_mean": mean, "diurnal_mean_std": mean_std,
        "diurnal_p99": p99, "diurnal_p99_std": p99_std,
        "exceedance": exceedance,
        "ensemble_spread": ensemble_spread,
    }


def _save_diurnal(hours_c, target_curve, model_curves, ylabel, title, out_path,
                  bands=None):
    plt.figure(figsize=(9, 5))
    plt.plot(hours_c, _cyclic(target_curve), "k--", lw=2.2, label="Target")
    for name, curve in model_curves.items():
        line, = plt.plot(hours_c, _cyclic(curve), lw=1.8, label=name)
        if bands and name in bands and bands[name] is not None:
            std = _cyclic(bands[name])
            mc = _cyclic(curve)
            plt.fill_between(hours_c, np.maximum(mc - std, 0), mc + std,
                             color=line.get_color(), alpha=0.18)
    plt.xlabel("Hour (UTC)")
    plt.xticks(range(0, 25, 3))
    plt.xlim(0, 24)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_path}")


def _save_exceedance(target_exc, model_excs, out_path):
    plt.figure(figsize=(9, 6))
    plt.plot(THRESHOLDS, target_exc, "k--", lw=2.2, label="Target")
    for name, exc in model_excs.items():
        plt.plot(THRESHOLDS, exc, lw=1.8, alpha=0.9, label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("All-hour Precipitation Over Land [mm/h] (Pooled, ensemble-mean)")
    plt.ylabel("Probability of Exceedance")
    plt.ylim(1e-8, 1)
    plt.xlim(THRESHOLDS[1], THRESHOLDS[-1])
    plt.title("Probability of Exceedance — model comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_path}")


def _rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


def _exceedance_log_err(model_exc, target_exc):
    """Mean |log10 P_model - log10 P_target| over resolvable target thresholds."""
    mask = (target_exc >= EXC_SCORE_FLOOR) & (target_exc <= 1.0)
    if not mask.any():
        return float("nan")
    eps = 1e-12
    diff = np.abs(np.log10(np.clip(model_exc, eps, None))
                  - np.log10(np.clip(target_exc, eps, None)))
    return float(np.mean(diff[mask]))


def main(cfg: dict, models_override=None, output_dir_override=None):
    logging.basicConfig(level=logging.INFO)

    models = models_override if models_override else cfg.get("models")
    if not models:
        logger.error("No models specified (config 'models' or --models).")
        return
    models = [{"name": m["name"], "dir": m["dir"]} for m in models]

    conv_factor = cfg.get("conv_factor", 24000)
    conv_factor_hourly = cfg.get("conv_factor_hourly", 1000)
    log_interval = cfg.get("log_interval", 24)
    output_dir = Path(output_dir_override or cfg.get("output_dir", "model_comparison"))

    # Channel indices + dataset come from the first model's generation config.
    first_dir = Path(models[0]["dir"])
    with open(first_dir / ".hydra" / "config.yaml") as f:
        gen_cfg = yaml.safe_load(f)
    dataset_cfg = gen_cfg["dataset"]
    dataset = known_datasets[dataset_cfg["type"]](**dataset_cfg)
    tp_out = get_channel_indices(dataset)["output"]["tp"]
    logger.info(f"tp output channel index: {tp_out}")

    # Land mask → boolean array (True over land).
    lm = load_land_sea_mask(cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width"))
    land_bool = ~np.isnan(lm.values)
    logger.info(f"Land pixels: {int(land_bool.sum())}")

    # The target reference is read once from `target_from` (targets are identical
    # across models and not duplicated into every model dir), so resolve it first.
    target_from = cfg.get("target_from", models[0]["name"])
    target_dir = next(m["dir"] for m in models if m["name"] == target_from)

    # Determine the common timestamp set: prediction models need only a
    # `-predictions` file; the common set is additionally restricted to timestamps
    # for which the target source has a `-target` file.
    avail = {m["name"]: _available_timestamps(Path(m["dir"])) for m in models}
    for name, ts_set in avail.items():
        logger.info(f"  {name}: {len(ts_set)} timesteps with predictions")
    target_avail = _available_timestamps(Path(target_dir), kind="target")
    logger.info(f"  [target:{target_from}] {len(target_avail)} timesteps with target")

    common = set.intersection(*avail.values()) if avail else set()
    common &= target_avail
    if cfg.get("times_range"):
        requested = set(get_time_from_range(cfg["times_range"], time_format="%Y%m%d-%H%M"))
        common &= requested
    times = sorted(common)
    if not times:
        logger.error("No common timestamps across all models — nothing to compare.")
        return
    logger.info(f"Comparing on {len(times)} common timestamps "
                f"({times[0]} … {times[-1]})")

    # Per-model summary cache: unchanged models are loaded instead of re-streamed.
    # The signature covers the exact common-timestep set, factors, land mask and
    # tensor mtimes, so any change forces a recompute of the affected model only.
    use_cache = cfg.get("cache", True)
    cache_dir = output_dir / "summary_cache"
    n_land = int(land_bool.sum())

    def summaries(name, model_dir, is_target_only):
        sig = _cache_signature(model_dir, times, conv_factor, conv_factor_hourly,
                               is_target_only, n_land)
        if use_cache:
            cached = _cache_load(cache_dir, name, sig)
            if cached is not None:
                return cached
        result = process_model(name, model_dir, times, tp_out, conv_factor,
                               conv_factor_hourly, land_bool, log_interval,
                               is_target_only=is_target_only)
        if use_cache:
            _cache_save(cache_dir, name, sig, result)
        return result

    # Target reference curves (single pass over -target files).
    logger.info(f"Computing target reference from '{target_from}'")
    tgt = summaries(target_from + "__target", target_dir, is_target_only=True)

    # Each model's ensemble curves.
    results = {}
    for m in models:
        logger.info(f"Processing model '{m['name']}'")
        results[m["name"]] = summaries(m["name"], m["dir"], is_target_only=False)

    # ---- Overlay figures ----
    hours_c = list(range(24)) + [24]
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_diurnal(hours_c, tgt["diurnal_mean"],
                  {n: r["diurnal_mean"] for n, r in results.items()},
                  "Precipitation (mm/day)", "Diurnal Cycle of Precip Amount — comparison",
                  output_dir / "diurnal_cycle_precip_amount_compare.png",
                  bands={n: r["diurnal_mean_std"] for n, r in results.items()})
    _save_diurnal(hours_c, tgt["diurnal_p99"],
                  {n: r["diurnal_p99"] for n, r in results.items()},
                  "Precipitation (mm/day)", "Diurnal Cycle of 99th-Percentile Precip — comparison",
                  output_dir / "diurnal_cycle_precip_99th_compare.png",
                  bands={n: r["diurnal_p99_std"] for n, r in results.items()})
    _save_exceedance(tgt["exceedance"],
                     {n: r["exceedance"] for n, r in results.items()},
                     output_dir / "precipitation_exceedance_compare.png")

    # ---- Scoreboard ----
    scoreboard = {}
    for name, r in results.items():
        scoreboard[name] = {
            "diurnal_mean_rmse": _rmse(r["diurnal_mean"], tgt["diurnal_mean"]),
            "diurnal_p99_rmse": _rmse(r["diurnal_p99"], tgt["diurnal_p99"]),
            "exceedance_log_err": _exceedance_log_err(r["exceedance"], tgt["exceedance"]),
            "ensemble_spread": r["ensemble_spread"],
        }
    # Only the vs-target metrics get a winner (lower=better). ensemble_spread is an
    # absolute per-pixel dispersion diagnostic (mm/h) — reported without a winner.
    metrics = ["diurnal_mean_rmse", "diurnal_p99_rmse", "exceedance_log_err"]
    winners = {met: min(scoreboard, key=lambda n: scoreboard[n][met]) for met in metrics}

    # Print a table (vs-target metrics with a winner star, then spread as info).
    cols = metrics + ["ensemble_spread"]
    header = f"{'model':<22}" + "".join(f"{met:>22}" for met in cols)
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    for name, sc in scoreboard.items():
        row = f"{name:<22}" + "".join(
            f"{sc[met]:>21.4f}" + ("*" if winners.get(met) == name else " ") for met in cols)
        logger.info(row)
    logger.info("=" * len(header))
    logger.info(f"(* = best per column; lower is better. ensemble_spread is mm/h, "
                f"informational. n_times={len(times)})")

    out_json = output_dir / "scoreboard.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_times": len(times),
            "times_range": [times[0], times[-1]],
            "target_from": target_from,
            "metrics": scoreboard,
            "winners": winners,
        }, f, indent=2)
    logger.info(f"Saved {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Path to comparison YAML config.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional name=dir overrides, e.g. corrdiff=/p/cd dit=/p/dit")
    parser.add_argument("--output-dir", default=None, help="Override output_dir.")
    args = parser.parse_args()

    with open(args.config_name) as f:
        cfg = yaml.safe_load(f)

    models_override = None
    if args.models:
        models_override = []
        for pair in args.models:
            name, _, d = pair.partition("=")
            models_override.append({"name": name, "dir": d})

    main(cfg, models_override=models_override, output_dir_override=args.output_dir)
