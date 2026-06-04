"""Shared machinery for the *bias / MAE / spread by percentile* plots."""
import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from hirad.eval.eval_utils import (
    get_channel_indices,
    load_generation_setup,
    load_land_sea_mask,
    resolve_ts_dir,
)

# Deterministic modes plotted alongside the ensemble, and the ensemble styling.
DET_MODE_CFG = [
    ('baseline',              'Input',                 'orange'),
    ('regression-prediction', 'Regression Prediction', 'red'),
]
ENSEMBLE_LABEL = 'CorrDiff Ensemble (mean +/- 1 sigma)'
ENSEMBLE_COLOR = 'green'

_RC_PARAMS = {
    'font.size':       16,
    'axes.titlesize':  18,
    'axes.labelsize':  16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
}


def to_flat(arr, conv: float, offset: float = 0.0) -> np.ndarray:
    """Convert a (possibly xarray) field to a flat float numpy array, scaled and shifted."""
    return (np.asarray(getattr(arr, 'values', arr)) * conv + offset).ravel()


def _smooth2d(arr, sigma: float) -> np.ndarray:
    """Apply an isotropic Gaussian low-pass to a 2D field (grid-point sigma)."""
    a = np.asarray(getattr(arr, 'values', arr), dtype=np.float32)
    return gaussian_filter(a, sigma=sigma, mode='nearest')


def build_all_histograms(
    times: list,
    ts_dirs: dict,
    out_channels: tuple,
    in_channels: tuple,
    reduce_fn: Callable,
    conv: float,
    offset: float,
    land_idx: np.ndarray,
    hist_bins: np.ndarray,
    log_interval: int,
    logger: logging.Logger,
    smoothing_sigma: float | None = None,
) -> tuple:
    """Single serial pass over all timesteps, building histograms for every mode."""
    n_land = land_idx.size
    n_bins = len(hist_bins) - 1
    det_modes = ('target', 'baseline', 'regression-prediction')
    interior_edges = hist_bins[1:-1]
    row_offsets = np.arange(n_land, dtype=np.intp) * n_bins

    det_counts: dict = {m: np.zeros((n_land, n_bins), dtype=np.int32) for m in det_modes}
    member_counts: list | None = None
    n_members: int | None = None
    skip_det: set = set()
    skip_preds = False

    def accumulate(counts, channel_arrs):
        if smoothing_sigma is not None:
            channel_arrs = [_smooth2d(a, smoothing_sigma) for a in channel_arrs]
        flats = [to_flat(a, conv, offset) for a in channel_arrs]
        vals = reduce_fn(flats)[land_idx]
        bin_idx = np.searchsorted(interior_edges, vals, side='right')
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        counts.reshape(-1)[row_offsets + bin_idx] += 1

    logger.info(f"Processing all modes in a single pass ({len(times)} timesteps)")

    for i, ts in enumerate(times):
        if i % log_interval == 0:
            logger.info(f"  timestep {i + 1}/{len(times)}")

        ts_dir = ts_dirs[ts]

        for mode in det_modes:
            if mode in skip_det:
                continue
            chans = in_channels if mode == 'baseline' else out_channels
            try:
                loaded = torch.load(ts_dir / f"{ts}-{mode}", weights_only=False)
            except FileNotFoundError:
                logger.warning(f"  [{mode}] file not found at {ts}, skipping mode")
                skip_det.add(mode)
                continue
            accumulate(det_counts[mode], [loaded[c] for c in chans])

        if not skip_preds:
            try:
                preds = torch.load(ts_dir / f"{ts}-predictions", weights_only=False)
            except FileNotFoundError:
                logger.warning(f"  [predictions] file not found at {ts}, skipping ensemble")
                skip_preds = True
                continue

            if n_members is None:
                n_members = int(preds.shape[0])
                member_counts = [
                    np.zeros((n_land, n_bins), dtype=np.int32)
                    for _ in range(n_members)
                ]
                logger.info(f"  Detected {n_members} ensemble members")
            assert n_members is not None and member_counts is not None
            for m in range(n_members):
                accumulate(member_counts[m], [preds[m, c] for c in out_channels])

    for mode in skip_det:
        det_counts[mode] = None

    return det_counts, member_counts, n_members


def per_point_quantiles(pp_counts: np.ndarray, bin_edges: np.ndarray,
                        frac_percentiles: np.ndarray,
                        block_size: int = 8192) -> np.ndarray:
    """Estimate per-row quantiles from per-grid-point histograms."""
    n_land, n_bins = pp_counts.shape
    P = len(frac_percentiles)
    result = np.empty((n_land, P), dtype=np.float32)
    edges_upper = bin_edges[1:].astype(np.float32)
    frac_f64 = frac_percentiles.astype(np.float64)

    for start in range(0, n_land, block_size):
        end = min(start + block_size, n_land)
        blk = pp_counts[start:end]
        B = end - start

        cdf = np.cumsum(blk, axis=1, dtype=np.float64)
        totals = cdf[:, -1:]
        cdf /= np.maximum(totals, 1.0)

        offset = (np.arange(B, dtype=np.float64) * 2.0)[:, None]
        cdf += offset
        queries = frac_f64[None, :] + offset

        idx = np.searchsorted(cdf.ravel(), queries.ravel(), side='left')
        idx = idx.reshape(B, P) - (np.arange(B, dtype=np.intp)[:, None] * n_bins)
        np.clip(idx, 0, n_bins - 1, out=idx)
        result[start:end] = edges_upper[idx]

    return result


def compute_quantiles(
    det_counts: dict,
    member_counts: list | None,
    n_members: int | None,
    active_det_modes: list,
    has_ensemble: bool,
    hist_bins: np.ndarray,
    frac_percentiles: np.ndarray,
    n_workers: int,
) -> tuple:
    """Compute per-point quantiles for every mode in parallel, freeing counts as we go."""
    members = member_counts if (has_ensemble and member_counts is not None and n_members is not None) else []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        def submit(counts):
            return pool.submit(per_point_quantiles, counts, hist_bins, frac_percentiles)

        fut_target = submit(det_counts['target'])
        fut_det = {mode: submit(det_counts[mode]) for mode in active_det_modes}
        fut_members = [submit(c) for c in members]

        target_q = fut_target.result()
        del det_counts['target']
        det_results = {mode: fut_det[mode].result() for mode in active_det_modes}
        for mode in active_det_modes:
            del det_counts[mode]
        member_qs = [f.result() for f in fut_members]
        members.clear()

    return target_q, det_results, member_qs


def _ensemble_stats(member_qs, target_q):
    """Compute ensemble spread, and the MAE / bias / FBI plot entries.

    All quantities use the same local-then-averaged estimator: a statistic
    is formed per grid point across ensemble members, then averaged over land.
    """
    n_members = len(member_qs)
    sum_q = np.zeros(target_q.shape, dtype=np.float64)
    sumsq_q = np.zeros(target_q.shape, dtype=np.float64)
    sum_ae = np.zeros(target_q.shape, dtype=np.float64)
    sumsq_ae = np.zeros(target_q.shape, dtype=np.float64)
    sum_fbi = np.zeros(target_q.shape, dtype=np.float64)
    sumsq_fbi = np.zeros(target_q.shape, dtype=np.float64)
    target_q_f = target_q.astype(np.float64)
    # FBI is a quantile ratio q_pred / q_target; guard against zero targets.
    target_q_safe = np.where(target_q_f != 0.0, target_q_f, np.nan)

    for qm in member_qs:
        qm_f = qm.astype(np.float64)
        sum_q += qm
        sumsq_q += qm_f ** 2
        ae = np.abs(qm_f - target_q_f)
        sum_ae += ae
        sumsq_ae += ae ** 2
        fbi = qm_f / target_q_safe
        sum_fbi += fbi
        sumsq_fbi += fbi ** 2

    mean_q = sum_q / n_members
    var_q = np.maximum(sumsq_q / n_members - mean_q ** 2, 0.0)
    std_q = np.sqrt(var_q)
    spread = std_q.mean(axis=0)

    mean_ae = sum_ae / n_members
    var_ae = np.maximum(sumsq_ae / n_members - mean_ae ** 2, 0.0)
    mae_entry = (mean_ae.mean(axis=0), np.sqrt(var_ae).mean(axis=0))

    # Bias band, local-then-averaged: per grid point the member-mean bias is
    # (mean_q - target_q) and the member-std of bias equals std_q (target is
    # constant across members); both are then averaged over land.
    bias_entry = ((mean_q - target_q_f).mean(axis=0), std_q.mean(axis=0))

    mean_fbi = sum_fbi / n_members
    var_fbi = np.maximum(sumsq_fbi / n_members - mean_fbi ** 2, 0.0)
    fbi_entry = (np.nanmean(mean_fbi, axis=0), np.nanmean(np.sqrt(var_fbi), axis=0))

    return spread, mae_entry, bias_entry, fbi_entry


def new_percentile_axes(percentile_values: np.ndarray):
    """Create a figure/axes pair and return it together with the fractional x-values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    return fig, ax, percentile_values / 100.0


def _round_sig(x: float, sig: int = 2) -> float:
    """Round *x* to *sig* significant figures (clean axis labels)."""
    if x == 0 or not np.isfinite(x):
        return 0.0
    digits = sig - int(np.floor(np.log10(abs(x)))) - 1
    return round(x, digits)


def even_value_ticks(frac: np.ndarray, mean_q: np.ndarray,
                     target_ticks: int = 9) -> tuple:
    """Pick secondary-axis ticks evenly spaced along the logit axis."""
    def _logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1.0 - p))

    def _expit(z):
        return 1.0 / (1.0 + np.exp(-z))

    v_lo, v_hi = float(mean_q[0]), float(mean_q[-1])
    if v_hi <= v_lo:
        return np.array([]), np.array([])

    lp_lo, lp_hi = _logit(frac[0]), _logit(frac[-1])
    # Evenly spaced sample positions along the logit axis (corners included).
    sample_lp = np.linspace(lp_lo, lp_hi, max(target_ticks, 2))
    sample_pos = _expit(sample_lp)
    sample_val = np.interp(sample_pos, frac, mean_q)

    # Round labels to two significant figures, keeping the true axis position.
    rounded = np.array([_round_sig(v) for v in sample_val])

    # Drop consecutive duplicate labels (can happen where the value saturates),
    # always keeping the first occurrence so both corners survive.
    keep = [0]
    for i in range(1, len(rounded)):
        if rounded[i] != rounded[keep[-1]]:
            keep.append(i)
    keep = np.array(keep, dtype=np.intp)
    return sample_pos[keep], rounded[keep]


def plot_dict_curves(ax, frac, data_dict, labels, colors, lower_clip=None) -> list:
    """Plot per-mode curves and return the arrays spanning the plotted range."""
    all_vals = []
    for (_key, data), label, color in zip(data_dict.items(), labels, colors):
        if isinstance(data, list):
            arr = np.array(data)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
        elif isinstance(data, tuple):
            mean, std = (np.asarray(data[0]), np.asarray(data[1]))
        else:
            ax.plot(frac, data, color=color, label=label, linewidth=2, alpha=0.85)
            all_vals.append(np.asarray(data))
            continue

        lower = mean - std if lower_clip is None else np.maximum(mean - std, lower_clip)
        upper = mean + std
        ax.plot(frac, mean, color=color, label=label, linewidth=2)
        ax.fill_between(frac, lower, upper, color=color, alpha=0.2)
        all_vals.extend([lower, upper])
    return all_vals


def finalize_percentile_plot(ax, frac, apply_xaxis, mean_q, xlabel, ylabel,
                             title, out_path, legend: bool = True) -> None:
    """Apply shared axis styling (via *apply_xaxis*) and write the figure to disk."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    apply_xaxis(ax, frac, mean_q)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


@dataclass
class BiasByPercentileSpec:
    """Variable-specific configuration for :func:`run_bias_by_percentile`."""
    var_label: str          # e.g. "2m temperature" — used in log messages
    output_prefix: str      # e.g. "temperature" — used in output file names
    bias_title: str
    mae_title: str
    spread_title: str
    fbi_title: str
    bias_ylabel: str
    mae_ylabel: str
    spread_ylabel: str
    fbi_ylabel: str
    percentile_values: np.ndarray
    resolve_channels: Callable[[dict], tuple]   # indices -> (ch_out, ch_in); raises ValueError
    make_hist_bins: Callable[[dict], np.ndarray]
    read_scaling: Callable[[dict], tuple]       # cfg -> (conv, offset)
    save_bias: Callable
    save_mae: Callable
    save_spread: Callable
    save_fbi: Callable
    # Combines the (scaled) per-channel flat fields into the plotted scalar.
    # Defaults to the single-channel identity; wind speed uses ``hypot``.
    reduce_fn: Callable[[list], np.ndarray] = lambda flats: flats[0]


def run_bias_by_percentile(cfg: dict, spec: BiasByPercentileSpec) -> None:
    """End-to-end driver shared by the temperature and precipitation scripts."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(spec.output_prefix)
    plt.rcParams.update(_RC_PARAMS)

    logger.info(f"Starting bias-by-percentile computation for {spec.var_label} over land")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    out_root = Path(generation_dir)

    indices = get_channel_indices(gen_cfg)
    try:
        ch_out, ch_in = spec.resolve_channels(indices)
    except ValueError as exc:
        logger.error(str(exc))
        return
    # Channels may be a single index or a tuple (e.g. wind u/v); normalize to tuples.
    out_channels = ch_out if isinstance(ch_out, tuple) else (ch_out,)
    in_channels = ch_in if isinstance(ch_in, tuple) else (ch_in,)
    logger.info(f"Channel indices - output: {out_channels}, input: {in_channels}")

    land_da = load_land_sea_mask(
        cfg.get("land_sea_mask_path"), cfg.get("height") or 352, cfg.get("width") or 544
    )
    land_idx = np.flatnonzero(np.isfinite(land_da.values).ravel())
    n_land = land_idx.size
    logger.info(f"{n_land} land grid points")

    hist_bins = spec.make_hist_bins(cfg)
    log_interval = cfg.get("log_interval", 24)
    conv, offset = spec.read_scaling(cfg)

    smoothing_km = cfg.get("smoothing_sigma_km")
    grid_res_km = cfg.get("grid_res_km", 1.0)
    smoothing_sigma = (smoothing_km / grid_res_km) if smoothing_km else None
    if smoothing_sigma is not None:
        logger.info(
            f"Gaussian smoothing enabled: sigma = {smoothing_km} km "
            f"/ {grid_res_km} km = {smoothing_sigma:.3g} grid points"
        )
    suffix = f"_smoothed{int(smoothing_km)}km" if smoothing_km else ""
    title_note = f" ({int(smoothing_km)} km smoothed)" if smoothing_km else ""

    percentile_values = spec.percentile_values
    frac_percentiles = percentile_values / 100.0

    logger.info(f"Resolving {len(times)} timestep directories ...")
    ts_dirs = {ts: resolve_ts_dir(out_root, ts) / ts for ts in times}

    det_counts, member_counts, n_members = build_all_histograms(
        times, ts_dirs, out_channels, in_channels, spec.reduce_fn, conv, offset,
        land_idx, hist_bins, log_interval, logger,
        smoothing_sigma=smoothing_sigma,
    )

    if det_counts.get('target') is None:
        logger.error("No target data found; cannot compute bias.")
        return

    active_det_modes = [m for m, _, _ in DET_MODE_CFG if det_counts.get(m) is not None]
    has_ensemble = member_counts is not None and n_members is not None and n_members > 0

    n_tasks = 1 + len(active_det_modes) + (n_members if has_ensemble else 0)
    n_quant_workers = cfg.get("n_quant_workers", n_tasks)

    target_q, det_results, member_qs = compute_quantiles(
        det_counts, member_counts, n_members, active_det_modes, has_ensemble,
        hist_bins, frac_percentiles, n_quant_workers,
    )

    target_mean_q = target_q.mean(axis=0)
    # FBI is a per-point quantile ratio q_pred / q_target; guard zero targets.
    target_q_safe = np.where(target_q != 0.0, target_q, np.nan)

    bias_data: dict = {}
    mae_data: dict = {}
    fbi_data: dict = {}
    labels: list = []
    colors: list = []

    for mode, label, color in DET_MODE_CFG:
        if mode not in det_results:
            continue
        pred_q = det_results.pop(mode)
        bias_data[mode] = pred_q.mean(axis=0) - target_mean_q
        mae_data[mode] = np.abs(pred_q - target_q).mean(axis=0)
        fbi_data[mode] = np.nanmean(pred_q / target_q_safe, axis=0)
        labels.append(label)
        colors.append(color)

    spread = None
    if has_ensemble:
        spread, mae_entry, bias_entry, fbi_entry = _ensemble_stats(
            member_qs, target_q,
        )
        bias_data['predictions'] = bias_entry
        mae_data['predictions'] = mae_entry
        fbi_data['predictions'] = fbi_entry
        labels.append(ENSEMBLE_LABEL)
        colors.append(ENSEMBLE_COLOR)

    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps") / "by_percentile"
    output_path.mkdir(parents=True, exist_ok=True)

    fn = output_path / f'{spec.output_prefix}_bias_by_percentile{suffix}.png'
    spec.save_bias(
        bias_data, percentile_values, labels, colors,
        title=spec.bias_title + title_note, xlabel='Percentile', ylabel=spec.bias_ylabel,
        out_path=fn, mean_q=target_mean_q,
    )
    logger.info(f"Bias-by-percentile plot saved: {fn}")

    fn_mae = output_path / f'{spec.output_prefix}_mae_by_percentile{suffix}.png'
    spec.save_mae(
        mae_data, percentile_values, labels, colors,
        title=spec.mae_title + title_note, xlabel='Percentile', ylabel=spec.mae_ylabel,
        out_path=fn_mae, mean_q=target_mean_q,
    )
    logger.info(f"MAE-by-percentile plot saved: {fn_mae}")

    fn_fbi = output_path / f'{spec.output_prefix}_fbi_by_percentile{suffix}.png'
    spec.save_fbi(
        fbi_data, percentile_values, labels, colors,
        title=spec.fbi_title + title_note, xlabel='Percentile', ylabel=spec.fbi_ylabel,
        out_path=fn_fbi, mean_q=target_mean_q,
    )
    logger.info(f"FBI-by-percentile plot saved: {fn_fbi}")

    if spread is not None:
        fn_spread = output_path / f'{spec.output_prefix}_spread_by_percentile{suffix}.png'
        spec.save_spread(
            spread, percentile_values,
            title=spec.spread_title + title_note, xlabel='Percentile', ylabel=spec.spread_ylabel,
            out_path=fn_spread, mean_q=target_mean_q,
        )
        logger.info(f"Spread-by-percentile plot saved: {fn_spread}")
