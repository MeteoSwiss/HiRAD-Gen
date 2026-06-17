"""
Plots bias / MAE / spread as a function of percentile for precipitation, using a
local-then-averaged estimator (see :mod:`hirad.eval.bias_by_percentile_common`).
"""
import numpy as np

from hirad.eval.bias_by_percentile_common import (
    LOGIT_PERCENTILE_TICKS_TAIL,
    BiasByPercentileSpec,
    apply_logit_percentile_xaxis,
    finalize_percentile_plot,
    new_percentile_axes,
    plot_dict_curves,
    run_bias_by_percentile,
)
from hirad.eval.eval_utils import make_percentile_values, parse_eval_cli


_PRECIP_SECONDARY_MMH = np.array([0.01, 0.1, 1.0, 10.0, 100.0])


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray | None = None) -> None:
    """Apply logit percentile x-axis (upper tail) with a mm/h secondary axis."""
    apply_logit_percentile_xaxis(
        ax, frac, mean_q,
        xlim_left=0.5,
        percentile_ticks=LOGIT_PERCENTILE_TICKS_TAIL,
        secondary_label='Mean target [mm/h]',
        secondary_values=_PRECIP_SECONDARY_MMH,
    )


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
        ymin = float(min(np.nanmin(v) for v in all_vals)) / 1.5
        ax.set_ylim(max(ymin, 1e-3), 10.0)
    finalize_percentile_plot(ax, frac, _apply_logit_xaxis,
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
    fbi_ylabel='FBI (exceedance)',
    percentile_values=make_percentile_values(),
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
