"""
Plots the bias (prediction - target) as a function of percentile for precipitation.

For each percentile level p, the bias is:

    bias(p) = quantile_pred(p) - quantile_target(p)

Positive bias means the model over-predicts at that quantile; negative means
under-prediction.  For ensemble predictions the per-member biases are averaged
and the ±1 sigma spread is shaded.
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

def _hist_quantiles(hist_counts, bin_edges, frac_percentiles):
    """Vectorised quantile estimation from a histogram via linear CDF interpolation.

    Parameters
    ----------
    hist_counts : (N,) int array
    bin_edges   : (N+1,) float array
    frac_percentiles : (P,) float array  - fractional values in [0, 1]

    Returns
    -------
    (P,) float array of estimated quantile values.
    """
    cdf = np.cumsum(hist_counts) / max(hist_counts.sum(), 1)
    return np.interp(frac_percentiles, cdf, bin_edges[1:])


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
    ax.set_ylim(-5, 5)
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
    ax.set_ylim(0, 5)
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
    ax.set_ylim(0, 5)
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

    logger.info("Starting bias-by-percentile computation for precipitation over land")
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
    logger.info(f"TP channel indices – output: {tp_out}, input: {tp_in}")

    # Land-sea mask
    land_mask = load_land_sea_mask(
        cfg.get("land_sea_mask_path"), cfg.get("height"), cfg.get("width")
    )

    # Fine-grained log-spaced histogram bins for accurate tail quantiles
    hist_bins = np.concatenate([
        np.array([0.0]),
        np.logspace(-2, 3.2, 5000),   # 0.01 → ~1585 mm/h
    ])
    n_hist_bins = len(hist_bins) - 1

    hist_counts: dict[str, np.ndarray] = {}
    totals: dict[str, int] = {}

    # -- Target and deterministic baselines --
    for mode in ['target', 'baseline', 'regression-prediction']:
        logger.info(f"Processing mode: {mode}")
        mode_hist = np.zeros(n_hist_bins, dtype=np.int64)
        mode_total = 0
        try:
            for i, ts in enumerate(times):
                if i % cfg.get("log_interval") == 0:
                    logger.info(f"  Timestep {i + 1}/{len(times)}")
                ch_idx = tp_out if mode in ('target', 'regression-prediction') else tp_in
                data = (
                    torch.load(
                        resolve_ts_dir(out_root, ts) / ts / f"{ts}-{mode}",
                        weights_only=False,
                    )[ch_idx]
                    * cfg.get("conv_factor_hourly")
                    * land_mask
                )
                land_values = data.values[~np.isnan(data.values)]
                mode_hist += np.histogram(land_values, bins=hist_bins)[0]
                mode_total += len(land_values)
        except Exception:
            logger.warning(f"  {mode} data not found, skipping")
            continue

        hist_counts[mode] = mode_hist
        totals[mode] = mode_total
        logger.info(f"  Processed {mode_total} land values for {mode}")

    # -- Ensemble predictions --
    logger.info("Processing predictions")
    n_members: int | None = None
    member_hist: list[np.ndarray] | None = None
    member_totals: list[int] | None = None

    for i, ts in enumerate(times):
        if i % cfg.get("log_interval") == 0:
            logger.info(f"  Timestep {i + 1}/{len(times)}")
        preds = (
            torch.load(
                resolve_ts_dir(out_root, ts) / ts / f"{ts}-predictions",
                weights_only=False,
            )
            * cfg.get("conv_factor_hourly")
        )  # shape: (n_members, n_channels, lat, lon)

        if n_members is None:
            n_members = preds.shape[0]
            member_hist = [np.zeros(n_hist_bins, dtype=np.int64) for _ in range(n_members)]
            member_totals = [0] * n_members

        for m in range(n_members):
            land_values = (preds[m, tp_out] * land_mask).values
            land_values = land_values[~np.isnan(land_values)]
            member_hist[m] += np.histogram(land_values, bins=hist_bins)[0]
            member_totals[m] += len(land_values)

    logger.info(f"Collected {n_members} ensemble members for predictions")

    # -- Build percentile grid --
    # Dense in the body, finer in the upper tail
    percentile_values = np.unique(np.concatenate([
        np.linspace(1.0, 90.0, 90),
        np.linspace(90.0, 99.0, 90),
        np.linspace(99.0, 99.9, 45),
        np.linspace(99.9, 99.99, 20),
    ]))
    frac_percentiles = percentile_values / 100.0

    # -- Compute target quantiles --
    if 'target' not in hist_counts:
        logger.error("No target data found; cannot compute bias.")
        return

    target_quantiles = _hist_quantiles(hist_counts['target'], hist_bins, frac_percentiles)

    # -- Compute biases --
    bias_data: dict = {}
    labels: list = []
    colors: list = []

    for mode, label, color in [
        ('baseline',              'Input',                'orange'),
        ('regression-prediction', 'Regression Prediction', 'red'),
    ]:
        if mode in hist_counts:
            bias_data[mode] = _hist_quantiles(hist_counts[mode], hist_bins, frac_percentiles) - target_quantiles
            labels.append(label)
            colors.append(color)

    member_biases = []
    if member_hist is not None and n_members > 0:
        member_quantiles = [
            _hist_quantiles(member_hist[m], hist_bins, frac_percentiles)
            for m in range(n_members)
        ]
        member_biases = [q - target_quantiles for q in member_quantiles]
        bias_data['predictions'] = member_biases
        labels.append('CorrDiff Ensemble (mean ± 1σ)')
        colors.append('green')

    output_path = out_root / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)

    # -- Bias plot --
    fn = output_path / 'precipitation_bias_by_percentile.png'
    save_bias_by_percentile_plot(
        bias_data,
        percentile_values,
        labels,
        colors,
        title='Precipitation Bias by Percentile - Over Land (Pooled Data)',
        xlabel='Percentile',
        ylabel='Bias (Pred − Target) [mm/h]',
        out_path=fn,
    )
    logger.info(f"Bias-by-percentile plot saved: {fn}")

    # -- MAE plot --
    mae_data: dict = {}
    mae_labels: list = []
    mae_colors: list = []
    for mode, label, color in [
        ('baseline',              'Input',                'orange'),
        ('regression-prediction', 'Regression Prediction', 'red'),
    ]:
        if mode in hist_counts:
            pred_q = _hist_quantiles(hist_counts[mode], hist_bins, frac_percentiles)
            mae_data[mode] = np.abs(pred_q - target_quantiles)
            mae_labels.append(label)
            mae_colors.append(color)
    if member_biases:
        mae_data['predictions'] = [np.abs(b) for b in member_biases]
        mae_labels.append('CorrDiff Ensemble (mean ± 1σ)')
        mae_colors.append('green')

    fn_mae = output_path / 'precipitation_mae_by_percentile.png'
    save_mae_by_percentile_plot(
        mae_data,
        percentile_values,
        mae_labels,
        mae_colors,
        title='Precipitation MAE by Percentile - Over Land (Pooled Data)',
        xlabel='Percentile',
        ylabel='MAE [mm/h]',
        out_path=fn_mae,
    )
    logger.info(f"MAE-by-percentile plot saved: {fn_mae}")

    # -- Ensemble spread plot --
    if member_biases:
        # std of member quantiles = std of member biases (target_quantiles is constant)
        spread = np.std(member_biases, axis=0)
        fn_spread = output_path / 'precipitation_spread_by_percentile.png'
        save_spread_by_percentile_plot(
            spread,
            percentile_values,
            title='Precipitation Ensemble Spread by Percentile - Over Land (Pooled Data)',
            xlabel='Percentile',
            ylabel='Spread (std of member quantiles) [mm/h]',
            out_path=fn_spread,
        )
        logger.info(f"Spread-by-percentile plot saved: {fn_spread}")


if __name__ == '__main__':
    main(parse_eval_cli())
