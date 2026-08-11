"""
Plots bias / MAE / spread as a function of percentile for 2m temperature, using a
local-then-averaged estimator (see :mod:`hirad.eval.bias_by_percentile_common`).
"""
import numpy as np

from hirad.eval.bias_by_percentile_common import (
    BiasByPercentileSpec,
    apply_logit_percentile_xaxis,
    finalize_percentile_plot,
    new_percentile_axes,
    plot_dict_curves,
    run_bias_by_percentile,
)
from hirad.eval.eval_utils import make_percentile_values, parse_eval_cli


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray | None = None) -> None:
    """Apply logit percentile x-axis with a °C secondary axis."""
    apply_logit_percentile_xaxis(ax, frac, mean_q, secondary_label='Mean target [°C]')


def save_bias_by_percentile_plot(bias_data_dict, percentile_values, labels, colors,
                                 title, xlabel, ylabel, out_path, mean_q=None,
                                 unit=None) -> None:
    """Save a bias-by-percentile figure (linear y-axis, data-driven limits)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    all_vals = plot_dict_curves(ax, frac, bias_data_dict, labels, colors)
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--')
    if all_vals:
        vcat = np.concatenate([np.asarray(v).ravel() for v in all_vals])
        vmin, vmax = float(np.nanmin(vcat)), float(np.nanmax(vcat))
        margin = max(abs(vmax - vmin) * 0.1, 0.05)
        ax.set_ylim(vmin - margin, vmax + margin)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def save_mae_by_percentile_plot(mae_data_dict, percentile_values, labels, colors,
                                title, xlabel, ylabel, out_path, mean_q=None,
                                unit=None) -> None:
    """Save a MAE-by-percentile figure (linear y-axis, data-driven limits)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    all_vals = plot_dict_curves(ax, frac, mae_data_dict, labels, colors, lower_clip=0)
    if all_vals:
        ymax = float(max(np.nanmax(v) for v in all_vals)) * 1.1
        if ymax > 0:
            ax.set_ylim(0, ymax)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def save_spread_by_percentile_plot(spread, percentile_values,
                                   title, xlabel, ylabel, out_path, mean_q=None,
                                   unit=None) -> None:
    """Save an ensemble-spread-by-percentile figure (linear y-axis)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    ax.plot(frac, spread, color='green', linewidth=2, label='Ensemble spread')
    ymax_spread = float(np.nanmax(spread)) * 1.1
    if ymax_spread > 0:
        ax.set_ylim(0, ymax_spread)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def _resolve_channels(indices: dict) -> tuple:
    # Temperature channel: try '2t' first, then 't2m'
    t2m_out = indices['output'].get('2t', indices['output'].get('t2m'))
    t2m_in = indices['input'].get('2t', indices['input'].get('t2m', t2m_out))
    if t2m_out is None:
        raise ValueError("Temperature channel (2t / t2m) not found in output channels.")
    return t2m_out, t2m_in


def _make_hist_bins(cfg: dict) -> np.ndarray:
    # Linear histogram bins in °C — fine enough to resolve sub-degree differences
    n_bins = cfg.get("n_bins", 2000)
    temp_min = cfg.get("temp_bin_min_celsius", -90.0)
    temp_max = cfg.get("temp_bin_max_celsius", 65.0)
    return np.linspace(temp_min, temp_max, n_bins + 1)


SPEC = BiasByPercentileSpec(
    var_label='2m temperature',
    output_prefix='temperature',
    bias_title='T2m Bias Over Land',
    mae_title='T2m MAE Over Land',
    spread_title='T2m Ensemble Spread Over Land',
    bias_ylabel='Bias [°C]',
    mae_ylabel='MAE [°C]',
    spread_ylabel='Ensemble Spread [°C]',
    percentile_values=make_percentile_values(),
    resolve_channels=_resolve_channels,
    make_hist_bins=_make_hist_bins,
    # Default: convert Kelvin → °C (conv=1.0, offset=-273.15)
    read_scaling=lambda cfg: (cfg.get("temp_conv_factor", 1.0),
                              cfg.get("temp_offset_celsius", -273.15)),
    save_bias=save_bias_by_percentile_plot,
    save_mae=save_mae_by_percentile_plot,
    save_spread=save_spread_by_percentile_plot,
)


def main(cfg: dict) -> None:
    run_bias_by_percentile(cfg, SPEC)


if __name__ == '__main__':
    main(parse_eval_cli())
