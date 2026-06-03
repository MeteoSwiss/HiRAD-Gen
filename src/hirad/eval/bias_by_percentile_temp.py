"""
Plots bias / MAE / spread as a function of percentile for 2m temperature, using a
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


def _apply_logit_xaxis(ax, frac: np.ndarray, mean_q: np.ndarray | None = None) -> None:
    """Apply logit x-axis with labelled percentile ticks (and a °C secondary axis)."""
    ax.set_xscale('logit')
    ax.set_xlim(frac[0], frac[-1])
    tick_fracs  = [0.0001, 0.001, 0.01, 0.1, 0.50, 0.90, 0.99, 0.999, 0.9999]
    tick_labels = ['0.01', '0.1', '1', '10', '50', '90', '99', '99.9', '99.99']
    # only show ticks within our data range
    valid_ticks = [(f, l) for f, l in zip(tick_fracs, tick_labels)
                   if frac[0] <= f <= frac[-1]]
    ax.set_xticks([f for f, _ in valid_ticks])
    ax.set_xticklabels([l for _, l in valid_ticks])
    ax.grid(True, alpha=0.3, which='both')

    if mean_q is not None:
        ax2 = ax.twiny()
        ax2.set_xscale('logit')
        ax2.set_xlim(frac[0], frac[-1])
        # Temperature is ~linear in percentile, but the axis is logit, so a fixed
        # coarse list of temps bunches near the median while the tails get no
        # labels. Use dense integer-degree candidates and greedily keep only those
        # spaced far enough apart *on the axis* (in logit units): the compressed
        # tails get fine 1° steps, the centre gets coarse steps, and the labels
        # end up evenly spaced.
        tick_positions, tick_temps = _even_temp_ticks(frac, mean_q)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{v:g}' for v in tick_temps])
        ax2.set_xlabel('Mean target [°C]')


def _even_temp_ticks(frac: np.ndarray, mean_q: np.ndarray,
                     min_gap_frac: float = 0.06) -> tuple:
    """Pick integer-degree temperature ticks evenly spaced along the logit axis.

    Candidates are every whole degree within the data range; we greedily keep a
    tick only if it is at least *min_gap_frac* of the axis span (measured in
    logit coordinates) from the previously kept one.  Returns ``(positions,
    temps)``.
    """
    def _logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1.0 - p))

    t_lo = int(np.ceil(mean_q[0]))
    t_hi = int(np.floor(mean_q[-1]))
    if t_hi <= t_lo:
        return np.array([]), np.array([])

    temps = np.arange(t_lo, t_hi + 1, dtype=float)
    positions = np.interp(temps, mean_q, frac)
    lp = _logit(positions)
    min_gap = abs(_logit(frac[-1]) - _logit(frac[0])) * min_gap_frac

    keep = [0]
    for i in range(1, len(lp)):
        if lp[i] - lp[keep[-1]] >= min_gap:
            keep.append(i)
    keep = np.array(keep, dtype=np.intp)
    return positions[keep], temps[keep]


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
    bias_title='2m Temperature Bias Over Land',
    mae_title='2m Temperature MAE Over Land',
    spread_title='2m Temperature Ensemble Spread Over Land',
    bias_ylabel='Bias [°C]',
    mae_ylabel='MAE [°C]',
    spread_ylabel='Ensemble Spread [°C]',
    percentile_values=np.unique(np.concatenate([
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.1, 1.0, 10),
        np.linspace(1.0, 10.0, 10),
        np.linspace(10.0, 90.0, 80),
        np.linspace(90.0, 99.0, 90),
        np.linspace(99.0, 99.9, 45),
        np.linspace(99.9, 99.99, 20),
    ])),
    mae_kind='member_list',
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
