"""
Plots bias / MAE / spread as a function of percentile for precipitation, using a
local-then-averaged estimator (see :mod:`hirad.eval.bias_by_percentile_common`).
"""
import numpy as np

from hirad.eval.bias_by_percentile_common import (
    BiasByPercentileSpec,
    finalize_percentile_plot,
    new_percentile_axes,
    plot_dict_curves,
    run_bias_by_percentile,
)
from hirad.eval.eval_utils import parse_eval_cli


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray | None = None,
                       xlim_left: float = 0.5) -> None:
    """Apply logit x-axis with labelled percentile ticks (and a mm/h secondary axis)."""
    ax.set_xscale('logit')
    ax.set_xlim(xlim_left, frac[-1])
    all_tick_fracs  = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 0.9999, 0.99999]
    all_tick_labels = ['10', '25', '50', '75', '90', '99', '99.9', '99.99', '99.999']
    valid_ticks = [(f, l) for f, l in zip(all_tick_fracs, all_tick_labels)
                   if xlim_left <= f <= frac[-1]]
    ax.set_xticks([f for f, _ in valid_ticks])
    ax.set_xticklabels([l for _, l in valid_ticks])
    ax.grid(True, alpha=0.3, which='both')

    if mean_q is not None:
        ax2 = ax.twiny()
        ax2.set_xscale('logit')
        ax2.set_xlim(xlim_left, frac[-1])
        nice_mmh = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        tick_positions = np.interp(nice_mmh, mean_q, frac)
        valid = (tick_positions > xlim_left) & (tick_positions < frac[-1])
        ax2.set_xticks(tick_positions[valid])
        ax2.set_xticklabels([f'{v:g}' for v in nice_mmh[valid]])
        ax2.set_xlabel('Mean target [mm/h]')


def save_bias_by_percentile_plot(bias_data_dict, percentile_values, labels, colors,
                                 title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save a bias-by-percentile figure (symlog y-axis)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    plot_dict_curves(ax, frac, bias_data_dict, labels, colors)
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yscale('symlog', linthresh=0.1, linscale=0.3)
    ax.set_ylim(-10, 10)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def save_mae_by_percentile_plot(mae_data_dict, percentile_values, labels, colors,
                                title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save a MAE-by-percentile figure (log y-axis)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    plot_dict_curves(ax, frac, mae_data_dict, labels, colors, lower_clip=0)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 100)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def save_spread_by_percentile_plot(spread, percentile_values,
                                   title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save an ensemble-spread-by-percentile figure (log y-axis, no legend)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    ax.plot(frac, spread, color='green', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 100)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path, legend=False)


def save_fbi_by_percentile_plot(fbi_data_dict, percentile_values, labels, colors,
                                title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save a frequency-bias-index-by-percentile figure (log y-axis, ratio around 1)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    all_vals = plot_dict_curves(ax, frac, fbi_data_dict, labels, colors, lower_clip=1e-3)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yscale('log')
    if all_vals:
        ymax = float(max(np.nanmax(v) for v in all_vals)) * 1.5
        ymin = float(min(np.nanmin(v) for v in all_vals)) / 1.5
        ax.set_ylim(max(ymin, 1e-3), max(ymax, 2.0))
    finalize_percentile_plot(ax, frac,
                             lambda ax_, frac_, mq: _apply_logit_xaxis(ax_, frac_, mq, xlim_left=0.10),
                             mean_q, xlabel, ylabel, title, out_path)


def _resolve_channels(indices: dict) -> tuple:
    tp_out = indices['output']['tp']
    tp_in = indices['input'].get('tp', tp_out)
    return tp_out, tp_in


def _make_hist_bins(cfg: dict) -> np.ndarray:
    n_bins = 500
    return np.concatenate([np.array([0.0]), np.logspace(-2, 3.2, n_bins)])


SPEC = BiasByPercentileSpec(
    var_label='precipitation',
    output_prefix='precipitation',
    bias_title='Precipitation Bias Over Land',
    mae_title='Precipitation MAE Over Land',
    spread_title='Precipitation Ensemble Spread Over Land',
    fbi_title='Precipitation Frequency Bias Index Over Land',
    bias_ylabel='Bias [mm/h]',
    mae_ylabel='MAE [mm/h]',
    spread_ylabel='Ensemble Spread [mm/h]',
    fbi_ylabel='FBI [-]',
    percentile_values=np.unique(np.concatenate([
        np.linspace(1.0, 90.0, 90),
        np.linspace(90.0, 99.0, 90),
        np.linspace(99.0, 99.9, 45),
        np.linspace(99.9, 99.99, 20),
        np.linspace(99.99, 99.999, 10),
    ])),
    resolve_channels=_resolve_channels,
    make_hist_bins=_make_hist_bins,
    read_scaling=lambda cfg: (cfg.get("conv_factor_hourly", 1.0), 0.0),
    save_bias=save_bias_by_percentile_plot,
    save_mae=save_mae_by_percentile_plot,
    save_spread=save_spread_by_percentile_plot,
    save_fbi=save_fbi_by_percentile_plot,
)


def main(cfg: dict) -> None:
    run_bias_by_percentile(cfg, SPEC)


if __name__ == '__main__':
    main(parse_eval_cli())
