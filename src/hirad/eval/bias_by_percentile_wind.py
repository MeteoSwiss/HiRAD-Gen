"""
Plots bias / MAE / spread as a function of percentile for 10 m wind speed, using a
local-then-averaged estimator (see :mod:`hirad.eval.bias_by_percentile_common`).

Wind speed is derived per grid point from the two surface wind components
(``10u``, ``10v``) as ``hypot(u, v)`` before histogramming.
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


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray | None = None,
                       xlim_left: float | None = None) -> None:
    """Apply logit percentile x-axis with an m/s secondary axis."""
    apply_logit_percentile_xaxis(ax, frac, mean_q, xlim_left=xlim_left,
                                 secondary_label='Mean target [m/s]')


def save_bias_by_percentile_plot(bias_data_dict, percentile_values, labels, colors,
                                 title, xlabel, ylabel, out_path, mean_q=None) -> None:
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
                                title, xlabel, ylabel, out_path, mean_q=None) -> None:
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
                                   title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save an ensemble-spread-by-percentile figure (linear y-axis)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    ax.plot(frac, spread, color='green', linewidth=2, label='Ensemble spread')
    ymax_spread = float(np.nanmax(spread)) * 1.1
    if ymax_spread > 0:
        ax.set_ylim(0, ymax_spread)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis, mean_q,
                             xlabel, ylabel, title, out_path)


def save_fbi_by_percentile_plot(fbi_data_dict, percentile_values, labels, colors,
                                title, xlabel, ylabel, out_path, mean_q=None) -> None:
    """Save a frequency-bias-index-by-percentile figure (log y-axis, ratio around 1)."""
    _, ax, frac = new_percentile_axes(percentile_values)
    all_vals = plot_dict_curves(ax, frac, fbi_data_dict, labels, colors, lower_clip=1e-3)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yscale('log')
    if all_vals:
        ymin = float(min(np.nanmin(v) for v in all_vals)) / 1.5
        ax.set_ylim(max(ymin, 1e-3), 10.0)
    finalize_percentile_plot(ax, frac,
                             lambda ax_, frac_, mq: _apply_logit_xaxis(ax_, frac_, mq, xlim_left=0.10),
                             mean_q, xlabel, ylabel, title, out_path)


def _resolve_channels(indices: dict) -> tuple:
    # Wind speed is derived from the two surface wind components.
    u_out = indices['output'].get('10u')
    v_out = indices['output'].get('10v')
    if u_out is None or v_out is None:
        raise ValueError("Wind components (10u / 10v) not found in output channels.")
    u_in = indices['input'].get('10u', u_out)
    v_in = indices['input'].get('10v', v_out)
    return (u_out, v_out), (u_in, v_in)


def _make_hist_bins(cfg: dict) -> np.ndarray:
    # Linear histogram bins in m/s — fine enough to resolve sub-m/s differences
    n_bins = cfg.get("wind_n_bins", 1500)
    speed_min = cfg.get("wind_bin_min_ms", 0.0)
    speed_max = cfg.get("wind_bin_max_ms", 75.0)
    return np.linspace(speed_min, speed_max, n_bins + 1)


SPEC = BiasByPercentileSpec(
    var_label='10 m wind speed',
    output_prefix='windspeed',
    bias_title='10 m Wind Speed Bias Over Land',
    mae_title='10 m Wind Speed MAE Over Land',
    spread_title='10 m Wind Speed Ensemble Spread Over Land',
    fbi_title='10 m Wind Speed Frequency Bias Index Over Land',
    bias_ylabel='Bias [m/s]',
    mae_ylabel='MAE [m/s]',
    spread_ylabel='Ensemble Spread [m/s]',
    fbi_ylabel='FBI (exceedance)',
    percentile_values=make_percentile_values(),
    resolve_channels=_resolve_channels,
    make_hist_bins=_make_hist_bins,
    read_scaling=lambda cfg: (cfg.get("wind_conv_factor", 1.0), 0.0),
    reduce_fn=lambda flats: np.hypot(flats[0], flats[1]),
    save_bias=save_bias_by_percentile_plot,
    save_mae=save_mae_by_percentile_plot,
    save_spread=save_spread_by_percentile_plot,
    save_fbi=save_fbi_by_percentile_plot,
)


def main(cfg: dict) -> None:
    run_bias_by_percentile(cfg, SPEC)


if __name__ == '__main__':
    main(parse_eval_cli())
