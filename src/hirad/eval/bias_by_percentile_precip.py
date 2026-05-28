"""
Plots bias / MAE / spread as a function of percentile for precipitation, using a
local-then-averaged estimator.

For each grid point g (and ensemble member m), a histogram of precipitation is
built over time and the per-percentile quantile q_{g,m}(p) is estimated.
Spatial / member averaging is then applied to produce the plotted curves:
"""
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _accumulate_per_point_hist(pp_counts: np.ndarray, values_land: np.ndarray,
                                bin_edges: np.ndarray, n_bins: int) -> None:
    """In-place add of one timestep of land values into per-grid-point histograms.

    pp_counts : (n_land, n_bins) int32  – modified in place.
    values_land : (n_land,) float       – one value per land grid point.
    """
    # searchsorted over the interior edges: result in [0, n_bins - 1]
    bin_idx = np.searchsorted(bin_edges[1:-1], values_land, side='right')
    n_land = pp_counts.shape[0]
    # Fancy indexing with unique indices is fully vectorised.
    pp_counts.reshape(-1)[np.arange(n_land) * n_bins + bin_idx] += 1


def _per_point_quantiles(pp_counts: np.ndarray, bin_edges: np.ndarray,
                          frac_percentiles: np.ndarray) -> np.ndarray:
    """Estimate per-row quantiles from per-grid-point histograms.

    Returns (n_land, P) float32 array.  Uses upper-bin-edge values (no in-bin
    interpolation) — adequate given fine log-spaced bins.
    """
    cdf = pp_counts.astype(np.float32, copy=True)
    np.cumsum(cdf, axis=1, out=cdf)
    totals = cdf[:, -1:].copy()
    cdf /= np.maximum(totals, 1.0)

    edges_upper = bin_edges[1:].astype(np.float32)
    n_land, n_bins = pp_counts.shape
    out = np.empty((n_land, len(frac_percentiles)), dtype=np.float32)

    # Reusable bool buffer to avoid repeated allocation.
    buf = np.empty(cdf.shape, dtype=bool)
    for j, p in enumerate(frac_percentiles):
        np.less(cdf, p, out=buf)
        idx = buf.sum(axis=1)                # first bin where cdf >= p
        np.clip(idx, 0, n_bins - 1, out=idx)
        out[:, j] = edges_upper[idx]
    return out


def _build_per_point_histogram(load_fn, times: list, land_idx: np.ndarray,
                                hist_bins: np.ndarray, log_interval: int,
                                logger: logging.Logger, mode_name: str
                                ) -> np.ndarray | None:
    """Stream timesteps through `load_fn(ts) -> (H*W,) float array` and return
    (n_land, n_bins) int32 per-grid-point histogram, or None on failure."""
    n_land = land_idx.size
    n_bins = len(hist_bins) - 1
    pp_counts = np.zeros((n_land, n_bins), dtype=np.int32)
    try:
        for i, ts in enumerate(times):
            if i % log_interval == 0:
                logger.info(f"  [{mode_name}] timestep {i + 1}/{len(times)}")
            flat = load_fn(ts)                       # (H*W,) float, no NaN
            _accumulate_per_point_hist(pp_counts, flat[land_idx], hist_bins, n_bins)
    except FileNotFoundError:
        logger.warning(f"  {mode_name} data not found, skipping")
        return None
    return pp_counts



# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def save_bias_by_percentile_plot(
    bias_data_dict: dict,
    percentile_values: np.ndarray,
    labels: list,
    colors: list,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path,
) -> None:
    """Save a bias-by-percentile figure.

    Parameters
    ----------
    bias_data_dict : dict mapping key → bias array (n_percentiles,) for single
        datasets, or list/tuple of (n_percentiles,) arrays for ensembles.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to fractions in (0, 1) for logit scale
    frac = percentile_values / 100.0

    for (key, bias_data), label, color in zip(bias_data_dict.items(), labels, colors):
        if isinstance(bias_data, (list, tuple)):
            # Ensemble: plot member average ± 1 σ
            arr = np.array(bias_data)   # (n_members, n_percentiles)
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

    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--', label='Zero bias')

    # Logit x-axis: compresses the centre and stretches both tails
    _apply_logit_xaxis(ax, frac)
    # Symlog: linear within ±linthresh, logarithmic beyond → "log away from zero"
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


def _apply_logit_xaxis(ax, frac: np.ndarray) -> None:
    """Apply logit x-axis with labelled percentile ticks."""
    ax.set_xscale('logit')
    ax.set_xlim(frac[0], frac[-1])
    tick_fracs  = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 0.9999]
    tick_labels = ['1',  '10', '25', '50', '75', '90', '99', '99.9', '99.99']
    ax.set_xticks(tick_fracs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both')


def save_mae_by_percentile_plot(
    mae_data_dict: dict,
    percentile_values: np.ndarray,
    labels: list,
    colors: list,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path,
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
            arr = np.array(mae_data)   # (n_members, n_percentiles), already absolute
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

    _apply_logit_xaxis(ax, frac)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 10)
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
) -> None:
    """Save an ensemble-spread-by-percentile figure.

    Spread is the inter-member standard deviation of the p-th quantile,
    i.e. how much the ensemble members disagree at each percentile level.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frac = percentile_values / 100.0

    ax.plot(frac, spread, color='green', linewidth=2, label='CorrDiff Ensemble')
    _apply_logit_xaxis(ax, frac)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting per-gridpoint bias-by-percentile computation for precipitation over land")
    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.info(f"Loaded {len(times)} timesteps to process")

    out_root = Path(generation_dir)

    # Channel indices
    indices = get_channel_indices(gen_cfg)
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Land-sea mask: build a boolean mask and a flat index list of land points
    land_da = load_land_sea_mask(
        cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width")
    )
    land_bool_2d = np.isfinite(land_da.values)                       # (H, W)
    land_idx = np.flatnonzero(land_bool_2d.ravel())                  # (n_land,)
    n_land = land_idx.size
    logger.info(f"{n_land} land grid points")

    # Log-spaced histogram bins.  Coarser than the pooled version since each
    # grid point only contributes ~ T samples (here T = {len(times)}).
    n_bins = 500
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-2, 3.2, n_bins),       # 0.01 -> ~1585 mm/h
    ])
    log_interval = cfg.get("log_interval", 24)
    conv = cfg.get("conv_factor_hourly")

    # -- Percentile grid (denser in the upper tail) --
    percentile_values = np.unique(np.concatenate([
        np.linspace(1.0, 90.0, 90),
        np.linspace(90.0, 99.0, 90),
        np.linspace(99.0, 99.9, 45),
        np.linspace(99.9, 99.99, 20),
    ]))
    frac_percentiles = percentile_values / 100.0
    P = len(frac_percentiles)

    # ---------------------------------------------------------------------
    # Loader helpers
    # ---------------------------------------------------------------------
    def _load_det(ts, mode):
        ch_idx = tp_out if mode in ('target', 'regression-prediction') else tp_in
        arr = torch.load(
            resolve_ts_dir(out_root, ts) / ts / f"{ts}-{mode}",
            weights_only=False,
        )[ch_idx]
        arr = np.asarray(getattr(arr, 'values', arr)) * conv
        return arr.ravel()

    def _load_member(ts, m, preds_cache):
        arr = preds_cache[m, tp_out]
        arr = np.asarray(getattr(arr, 'values', arr)) * conv
        return arr.ravel()

    # ---------------------------------------------------------------------
    # Target: per-grid-point quantiles (needed by everything else)
    # ---------------------------------------------------------------------
    logger.info("Processing target")
    target_pp_counts = _build_per_point_histogram(
        lambda ts: _load_det(ts, 'target'),
        times, land_idx, hist_bins, log_interval, logger, 'target',
    )
    if target_pp_counts is None:
        logger.error("No target data found; cannot compute bias.")
        return
    target_q = _per_point_quantiles(target_pp_counts, hist_bins, frac_percentiles)
    del target_pp_counts
    target_mean_q = target_q.mean(axis=0)

    # ---------------------------------------------------------------------
    # Deterministic modes
    # ---------------------------------------------------------------------
    bias_data: dict = {}
    mae_data: dict = {}
    labels: list = []
    colors: list = []
    mae_labels: list = []
    mae_colors: list = []

    for mode, label, color in [
        ('baseline',              'Input',                'orange'),
        ('regression-prediction', 'Regression Prediction', 'red'),
    ]:
        logger.info(f"Processing {mode}")
        pp = _build_per_point_histogram(
            lambda ts, m=mode: _load_det(ts, m),
            times, land_idx, hist_bins, log_interval, logger, mode,
        )
        if pp is None:
            continue
        pred_q = _per_point_quantiles(pp, hist_bins, frac_percentiles)
        del pp
        bias_data[mode] = pred_q.mean(axis=0) - target_mean_q
        mae_data[mode] = np.abs(pred_q - target_q).mean(axis=0)
        labels.append(label)
        colors.append(color)
        mae_labels.append(label)
        mae_colors.append(color)

    # ---------------------------------------------------------------------
    # Ensemble predictions: process one timestep at a time, accumulating
    # per-member per-grid-point histograms.  Then collapse per-member to
    # spatial-mean curves and online-aggregate ensemble statistics.
    # ---------------------------------------------------------------------
    logger.info("Processing predictions (per-member, per-grid-point)")
    n_members: int | None = None
    member_pp: list[np.ndarray] | None = None

    for i, ts in enumerate(times):
        if i % log_interval == 0:
            logger.info(f"  [predictions] timestep {i + 1}/{len(times)}")
        preds = torch.load(
            resolve_ts_dir(out_root, ts) / ts / f"{ts}-predictions",
            weights_only=False,
        )  # (n_members, n_channels, H, W)
        if n_members is None:
            n_members = preds.shape[0]
            member_pp = [
                np.zeros((n_land, n_bins), dtype=np.int32) for _ in range(n_members)
            ]
            logger.info(f"  Detected {n_members} ensemble members")
        for m in range(n_members):
            arr = preds[m, tp_out]
            flat = (np.asarray(getattr(arr, 'values', arr)) * conv).ravel()
            _accumulate_per_point_hist(member_pp[m], flat[land_idx], hist_bins, n_bins)

    member_biases: list[np.ndarray] = []
    member_maes: list[np.ndarray] = []
    # Online aggregates for spread:  E[std_m(q_{g,m})] over g.
    # Need per-(g, p) std across members -> keep running sum and sum-of-squares
    # of per-member per-point quantiles.
    if member_pp is not None and n_members is not None and n_members > 0:
        sum_q = np.zeros((n_land, P), dtype=np.float64)
        sumsq_q = np.zeros((n_land, P), dtype=np.float64)
        for m in range(n_members):
            qm = _per_point_quantiles(member_pp[m], hist_bins, frac_percentiles)
            member_pp[m] = None  # free as we go
            sum_q += qm
            sumsq_q += qm.astype(np.float64) ** 2
            member_biases.append((qm.mean(axis=0) - target_mean_q).astype(np.float64))
            member_maes.append(np.abs(qm - target_q).mean(axis=0).astype(np.float64))
        mean_q = sum_q / n_members
        var_q = np.maximum(sumsq_q / n_members - mean_q ** 2, 0.0)
        # Per-gridpoint std across members, then spatial mean.
        spread = np.sqrt(var_q).mean(axis=0)

        bias_data['predictions'] = member_biases
        mae_data['predictions'] = member_maes
        labels.append('CorrDiff Ensemble (mean +/- 1 sigma)')
        colors.append('green')
        mae_labels.append('CorrDiff Ensemble (mean +/- 1 sigma)')
        mae_colors.append('green')
    else:
        spread = None

    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------
    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps") / "by_percentile"
    output_path.mkdir(parents=True, exist_ok=True)

    fn = output_path / 'precipitation_bias_by_percentile.png'
    save_bias_by_percentile_plot(
        bias_data, percentile_values, labels, colors,
        title='Precipitation Bias by Percentile - Over Land (Per-Gridpoint)',
        xlabel='Percentile',
        ylabel='Bias [mm/h]',
        out_path=fn,
    )
    logger.info(f"Bias-by-percentile plot saved: {fn}")

    fn_mae = output_path / 'precipitation_mae_by_percentile.png'
    save_mae_by_percentile_plot(
        mae_data, percentile_values, mae_labels, mae_colors,
        title='Precipitation MAE by Percentile - Over Land (Per-Gridpoint)',
        xlabel='Percentile',
        ylabel='MAE [mm/h]',
        out_path=fn_mae,
    )
    logger.info(f"MAE-by-percentile plot saved: {fn_mae}")

    if spread is not None:
        fn_spread = output_path / 'precipitation_spread_by_percentile.png'
        save_spread_by_percentile_plot(
            spread, percentile_values,
            title='Precipitation Ensemble Spread by Percentile - Over Land (Per-Gridpoint)',
            xlabel='Percentile',
            ylabel='Spread (mean over land of std across members) [mm/h]',
            out_path=fn_spread,
        )
        logger.info(f"Spread-by-percentile plot saved: {fn_spread}")


if __name__ == '__main__':
    main(parse_eval_cli())
