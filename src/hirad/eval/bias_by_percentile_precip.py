"""
Plots bias / MAE / spread as a function of percentile for precipitation, using a
local-then-averaged estimator.

For each grid point g (and ensemble member m), a histogram of precipitation is
built over time and the per-percentile quantile q_{g,m}(p) is estimated.
Spatial / member averaging is then applied to produce the plotted curves.
"""
import concurrent.futures
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hirad.eval.eval_utils import (
    get_channel_indices,
    load_generation_setup,
    load_land_sea_mask,
    parse_eval_cli,
    resolve_ts_dir,
)


def _to_flat(arr, conv: float) -> np.ndarray:
    """Convert a (possibly xarray) field to a flat float numpy array scaled by conv."""
    return (np.asarray(getattr(arr, 'values', arr)) * conv).ravel()


def _build_all_histograms(
    times: list,
    ts_dirs: dict,
    tp_out: int,
    tp_in: int,
    conv: float,
    land_idx: np.ndarray,
    hist_bins: np.ndarray,
    log_interval: int,
    logger: logging.Logger,
) -> tuple:
    """Single serial pass over all timesteps, building histograms for every mode.

    Returns ``(det_counts, member_counts, n_members)`` where *det_counts* maps
    mode → array-or-None and *member_counts* is a list of arrays (one per member)
    or None.
    """
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

    def accumulate(counts, arr):
        vals = _to_flat(arr, conv)[land_idx]
        bin_idx = np.searchsorted(interior_edges, vals, side='right')
        counts.reshape(-1)[row_offsets + bin_idx] += 1

    logger.info(f"Processing all modes in a single pass ({len(times)} timesteps)")

    for i, ts in enumerate(times):
        if i % log_interval == 0:
            logger.info(f"  timestep {i + 1}/{len(times)}")

        ts_dir = ts_dirs[ts]

        for mode in det_modes:
            if mode in skip_det:
                continue
            ch = tp_in if mode == 'baseline' else tp_out
            try:
                arr = torch.load(ts_dir / f"{ts}-{mode}", weights_only=False)[ch]
            except FileNotFoundError:
                logger.warning(f"  [{mode}] file not found at {ts}, skipping mode")
                skip_det.add(mode)
                continue
            accumulate(det_counts[mode], arr)

        if not skip_preds:
            try:
                preds = torch.load(ts_dir / f"{ts}-predictions", weights_only=False)
            except FileNotFoundError:
                logger.warning(f"  [predictions] file not found at {ts}, skipping ensemble")
                skip_preds = True
                continue

            if n_members is None:
                n_members = preds.shape[0]
                member_counts = [
                    np.zeros((n_land, n_bins), dtype=np.int32)
                    for _ in range(n_members)
                ]
                logger.info(f"  Detected {n_members} ensemble members")
            for m in range(n_members):
                accumulate(member_counts[m], preds[m, tp_out])

    for mode in skip_det:
        det_counts[mode] = None

    return det_counts, member_counts, n_members


def _per_point_quantiles(pp_counts: np.ndarray, bin_edges: np.ndarray,
                          frac_percentiles: np.ndarray,
                          block_size: int = 8192) -> np.ndarray:
    """Estimate per-row quantiles from per-grid-point histograms.

    Returns (n_land, P) float32 array.  Uses upper-bin-edge values (no in-bin
    interpolation) — adequate given fine log-spaced bins.

    Processes land points in blocks of *block_size* rows so the CDF working set
    (~block_size × n_bins × 8 bytes) fits comfortably in L3 cache, making the
    row-wise ``searchsorted`` cache-friendly and GIL-free (numpy releases the
    GIL for large C-level operations, enabling true thread parallelism when
    multiple calls run concurrently).
    """
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


def save_bias_by_percentile_plot(
    bias_data_dict: dict,
    percentile_values: np.ndarray,
    labels: list,
    colors: list,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path,
    mean_q: np.ndarray = None,
) -> None:
    """Save a bias-by-percentile figure.

    Parameters
    ----------
    bias_data_dict : dict mapping key → bias array (n_percentiles,) for single
        datasets, or list/tuple of (n_percentiles,) arrays for ensembles.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    frac = percentile_values / 100.0

    for (key, bias_data), label, color in zip(bias_data_dict.items(), labels, colors):
        if isinstance(bias_data, (list, tuple)):
            arr = np.array(bias_data)
            mean_bias = arr.mean(axis=0)
            std_bias = arr.std(axis=0)
            ax.plot(frac, mean_bias, color=color, label=label, linewidth=2)
            ax.fill_between(
                frac,
                mean_bias - std_bias,
                mean_bias + std_bias,
                color=color,
                alpha=0.2,
            )
        else:
            ax.plot(
                frac, bias_data,
                color=color, label=label, linewidth=2, alpha=0.85,
            )

    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    _apply_logit_xaxis(ax, frac, mean_q)
    ax.set_yscale('symlog', linthresh=0.1, linscale=0.3)
    ax.set_ylim(-10, 10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray = None) -> None:
    """Apply logit x-axis with labelled percentile ticks."""
    ax.set_xscale('logit')
    ax.set_xlim(0.5, frac[-1])
    tick_fracs  = [0.50, 0.75, 0.90, 0.99, 0.999, 0.9999]
    tick_labels = ['50', '75', '90', '99', '99.9', '99.99']
    ax.set_xticks(tick_fracs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both')

    if mean_q is not None:
        ax2 = ax.twiny()
        ax2.set_xscale('logit')
        ax2.set_xlim(0.5, frac[-1])
        # Place ticks at fixed "nice" mm/h values, positioning them by inverting mean_q
        nice_mmh = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        tick_positions = np.interp(nice_mmh, mean_q, frac)
        valid = (tick_positions > 0.5) & (tick_positions < frac[-1])
        tick_positions = tick_positions[valid]
        tick_mmh = nice_mmh[valid]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{v:g}' for v in tick_mmh])
        ax2.set_xlabel('Mean target [mm/h]')


def save_mae_by_percentile_plot(
    mae_data_dict: dict,
    percentile_values: np.ndarray,
    labels: list,
    colors: list,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path,
    mean_q: np.ndarray = None,
) -> None:
    """Save a MAE-by-percentile figure.

    For single datasets the MAE curve is plotted directly.  For the ensemble
    the mean absolute error across members is shown with ±1 σ shading.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frac = percentile_values / 100.0

    for (key, mae_data), label, color in zip(mae_data_dict.items(), labels, colors):
        if isinstance(mae_data, (list, tuple)):
            arr = np.array(mae_data)
            mean_mae = arr.mean(axis=0)
            std_mae = arr.std(axis=0)
            ax.plot(frac, mean_mae, color=color, label=label, linewidth=2)
            ax.fill_between(
                frac,
                np.maximum(mean_mae - std_mae, 0),
                mean_mae + std_mae,
                color=color, alpha=0.2,
            )
        else:
            ax.plot(frac, mae_data, color=color, label=label, linewidth=2, alpha=0.85)

    _apply_logit_xaxis(ax, frac, mean_q)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_spread_by_percentile_plot(
    spread: np.ndarray,
    percentile_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path,
    mean_q: np.ndarray = None,
) -> None:
    """Save an ensemble-spread-by-percentile figure.

    Spread is the inter-member standard deviation of the p-th quantile,
    i.e. how much the ensemble members disagree at each percentile level.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frac = percentile_values / 100.0

    ax.plot(frac, spread, color='green', linewidth=2)
    _apply_logit_xaxis(ax, frac, mean_q)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    plt.rcParams.update({
        'font.size':        16,
        'axes.titlesize':   18,
        'axes.labelsize':   16,
        'xtick.labelsize':  14,
        'ytick.labelsize':  14,
        'legend.fontsize':  14,
    })

    logger.info("Starting bias-by-percentile computation for precipitation over land")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    out_root = Path(generation_dir)

    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    land_da = load_land_sea_mask(
        cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width")
    )
    land_bool_2d = np.isfinite(land_da.values)
    land_idx = np.flatnonzero(land_bool_2d.ravel())
    n_land = land_idx.size
    logger.info(f"{n_land} land grid points")

    n_bins = 500
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-2, 3.2, n_bins),
    ])
    log_interval = cfg.get("log_interval", 24)
    conv = cfg.get("conv_factor_hourly", 1.0)

    percentile_values = np.unique(np.concatenate([
        np.linspace(1.0, 90.0, 90),
        np.linspace(90.0, 99.0, 90),
        np.linspace(99.0, 99.9, 45),
        np.linspace(99.9, 99.99, 20),
    ]))
    frac_percentiles = percentile_values / 100.0
    P = len(frac_percentiles)

    logger.info(f"Resolving {len(times)} timestep directories ...")
    ts_dirs = {ts: resolve_ts_dir(out_root, ts) / ts for ts in times}

    det_counts, member_counts, n_members = _build_all_histograms(
        times, ts_dirs, tp_out, tp_in, conv, land_idx, hist_bins,
        log_interval, logger,
    )

    if det_counts.get('target') is None:
        logger.error("No target data found; cannot compute bias.")
        return

    det_mode_cfg = [
        ('baseline',              'Input',                'orange'),
        ('regression-prediction', 'Regression Prediction', 'red'),
    ]
    active_det_modes = [m for m, _, _ in det_mode_cfg if det_counts.get(m) is not None]
    has_ensemble = member_counts is not None and n_members is not None and n_members > 0

    n_tasks = 1 + len(active_det_modes) + (n_members if has_ensemble else 0)
    n_quant_workers = cfg.get("n_quant_workers", n_tasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_quant_workers) as pool:
        def submit(counts):
            return pool.submit(_per_point_quantiles, counts, hist_bins, frac_percentiles)

        fut_target = submit(det_counts['target'])
        fut_det = {mode: submit(det_counts[mode]) for mode in active_det_modes}
        fut_members = (
            [submit(member_counts[m]) for m in range(n_members)]
            if has_ensemble else []
        )

        target_q = fut_target.result()
        del det_counts['target']
        det_results = {mode: fut_det[mode].result() for mode in active_det_modes}
        for mode in active_det_modes:
            del det_counts[mode]
        member_qs = [f.result() for f in fut_members]
        if has_ensemble:
            for m in range(n_members):
                member_counts[m] = None

    target_mean_q = target_q.mean(axis=0)

    bias_data: dict = {}
    mae_data: dict = {}
    labels: list = []
    colors: list = []

    for mode, label, color in det_mode_cfg:
        if mode not in det_results:
            continue
        pred_q = det_results.pop(mode)
        bias_data[mode] = pred_q.mean(axis=0) - target_mean_q
        mae_data[mode] = np.abs(pred_q - target_q).mean(axis=0)
        labels.append(label)
        colors.append(color)

    member_biases: list[np.ndarray] = []
    member_maes: list[np.ndarray] = []
    spread = None

    if has_ensemble:
        sum_q = np.zeros((n_land, P), dtype=np.float64)
        sumsq_q = np.zeros((n_land, P), dtype=np.float64)
        for qm in member_qs:
            sum_q += qm
            sumsq_q += qm.astype(np.float64) ** 2
            member_biases.append((qm.mean(axis=0) - target_mean_q).astype(np.float64))
            member_maes.append(np.abs(qm - target_q).mean(axis=0).astype(np.float64))
        mean_q = sum_q / n_members
        var_q = np.maximum(sumsq_q / n_members - mean_q ** 2, 0.0)
        spread = np.sqrt(var_q).mean(axis=0)

        bias_data['predictions'] = member_biases
        mae_data['predictions'] = member_maes
        labels.append('CorrDiff Ensemble (mean +/- 1 sigma)')
        colors.append('green')

    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps") / "by_percentile"
    output_path.mkdir(parents=True, exist_ok=True)

    fn = output_path / 'precipitation_bias_by_percentile.png'
    save_bias_by_percentile_plot(
        bias_data, percentile_values, labels, colors,
        title='Precipitation Bias Over Land',
        xlabel='Percentile',
        ylabel='Bias [mm/h]',
        out_path=fn,
        mean_q=target_mean_q,
    )
    logger.info(f"Bias-by-percentile plot saved: {fn}")

    fn_mae = output_path / 'precipitation_mae_by_percentile.png'
    save_mae_by_percentile_plot(
        mae_data, percentile_values, labels, colors,
        title='Precipitation MAE Over Land',
        xlabel='Percentile',
        ylabel='MAE [mm/h]',
        out_path=fn_mae,
        mean_q=target_mean_q,
    )
    logger.info(f"MAE-by-percentile plot saved: {fn_mae}")

    if spread is not None:
        fn_spread = output_path / 'precipitation_spread_by_percentile.png'
        save_spread_by_percentile_plot(
            spread, percentile_values,
            title='Precipitation Ensemble Spread Over Land',
            xlabel='Percentile',
            ylabel='Ensemble Spread [mm/h]',
            out_path=fn_spread,
            mean_q=target_mean_q,
        )
        logger.info(f"Spread-by-percentile plot saved: {fn_spread}")



if __name__ == '__main__':
    main(parse_eval_cli())
