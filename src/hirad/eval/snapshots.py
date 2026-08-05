"""Generates maps of precipitation, temperature, and wind components/speed/direction."""
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch

from hirad.datasets import get_channels_from_strings, get_strings_from_channels
from hirad.eval import compute_mae, plot_map
from hirad.eval.eval_utils import (
    DEFAULT_GRID_CONFIG,
    grid_cfg_from_cfg,
    load_generation_setup,
    parse_eval_cli,
    resolve_io_channels,
    resolve_ts_dir,
)
from hirad.eval.plotting import plot_map_precipitation, plot_map_wind_precip
from hirad.utils.inference_utils import calculate_bounds


def wind_direction(u, v):
    """Compute wind direction from u and v components."""
    return (np.arctan2(-u, -v) * 180 / np.pi) % 360

@dataclass
class ChannelMeta:
    """Metadata for a channel."""
    name:       str
    cmap:       str        = "viridis"
    me_cmap:    str | None = None
    unit:       str        = ""
    norm:       any        = None
    err_vmin:   float      = None
    err_vmax:   float      = None
    vmin:       float      = None
    vmax:       float      = None
    extend:     str        = "both"
    precip_kwargs: dict    = field(default_factory=lambda: {"threshold": 0.01, "rfac": 1000.0})

    @classmethod
    def get(cls, ch_or_name: "ChannelMeta | str | None", *, vmin=None, vmax=None) -> "ChannelMeta":
        name = getattr(ch_or_name, "name", ch_or_name or "")
        base = CHANNELS.get(name) or cls(name=name)
        if vmin is not None or vmax is not None:
            return replace(base, vmin=vmin, vmax=vmax)
        return base

CHANNELS = {
    "tp": ChannelMeta(name="tp", cmap=None, unit="mm/h", extend="max", precip_kwargs={"threshold": 0.01, "rfac": 1000.0}),
    "2t": ChannelMeta(name="2t", cmap="RdYlBu_r", me_cmap="RdBu", unit="K", err_vmin=-4.5, err_vmax=4.5),
    "10u": ChannelMeta(name="10u", cmap="BrBG", me_cmap="BrBG", unit="m/s", err_vmin=-10, err_vmax=10, vmin=-10, vmax=10),
    "10v": ChannelMeta(name="10v", cmap="BrBG", me_cmap="BrBG", unit="m/s", err_vmin=-10, err_vmax=10, vmin=-10, vmax=10),
}

def format_time_str(dt_str, input_fmt="%Y%m%d-%H%M", output_fmt="%d-%m-%Y %H:%M"):
    """Convert time string from input_fmt to output_fmt."""
    dt = datetime.strptime(dt_str, input_fmt)
    return dt.strftime(output_fmt)

class FileRepository:
    def __init__(self, root_path):
        self.root = Path(root_path)

    def load(self, time, filename):
        return torch.load(resolve_ts_dir(self.root, time) / time / filename, weights_only=False)

    def _ensure_dir(self, *subdirs):
        """Make (and return) root_path/subdir1/subdir2/…."""
        d = self.root.joinpath(*subdirs)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_fname(self, curr_time, prefix, suffix, member_idx):
        """Build a filename like '20250724-1230-prefix-suffix[_member]'."""
        base = f"{curr_time}-{prefix}-{suffix}"
        if member_idx is not None:
            base += f"_{member_idx}"
        return base

    def output_file(self, channel, curr_time, suffix, member_idx=None):
        # decide on the folder name: e.g. 'tp_100m' or just 'tp'
        folder = f"{channel.name}_{channel.level}" if getattr(channel, "level", None) else channel.name
        fname  = self._make_fname(curr_time, channel.name, suffix, member_idx)
        return self._ensure_dir(folder) / fname

    def wind_file(self, wind_type, curr_time, suffix, member_idx=None):
        # e.g. wind_type = "FF10m" or "DD10m"
        fname = self._make_fname(curr_time, wind_type, suffix, member_idx)
        return self._ensure_dir(wind_type) / fname


def _pred_members(arr, *channel_idxs):
    """Yield ``(member_idx_or_None, *2d_slices)`` for 3D or 4D ensemble arrays."""
    if arr.ndim > 3:
        for m in range(arr.shape[0]):
            yield (m, *(arr[m, i, :, :] for i in channel_idxs))
    else:
        yield (None, *(arr[i, :, :] for i in channel_idxs))


def save_field(name, data, meta, output_files, channel, t, member=None, kind=None, cmap=None, vmin=None, vmax=None, custom_path=None, plot_func=None, title=None, grid_cfg=DEFAULT_GRID_CONFIG, **plot_kwargs):
    """Save a field by plotting it with the appropriate function and parameters. Handles precipitation specially and supports custom paths."""
    # Determine output path
    suffix = f"{name}-{kind}" if kind else f"{name}"   
    out_path = custom_path or output_files.output_file(channel, t, suffix, member)

    # Choose plotting function
    plot = plot_func or plot_map

    # Case dependent plot parameters
    title = title or meta.name
    extend = 'max' if kind == 'mae' else meta.extend
    plot_map_args = {'title': title, 'norm': meta.norm, 'extend': extend, **plot_kwargs}
    common_args = {'grid_cfg': grid_cfg}

    # Precipitation case
    if plot.__name__ == 'plot_map_precipitation':
        precip_args = {**meta.precip_kwargs, **{k: plot_kwargs[k] for k in meta.precip_kwargs if k in plot_kwargs}}
        plot(data, out_path, title=title, **precip_args, **common_args)
    else:
        plot(data, out_path, vmin=vmin if vmin is not None else meta.vmin, vmax=vmax if vmax is not None else meta.vmax, cmap=cmap or meta.cmap, label=meta.unit, **plot_map_args, **common_args)


def main(cfg: dict) -> None:
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("plot_maps")

    grid_cfg = grid_cfg_from_cfg(cfg)

    try:
        generation_dir, gen_cfg, times = load_generation_setup(cfg)
    except ValueError as exc:
        logger.error(str(exc))
        return

    input_channels, output_channels = resolve_io_channels(gen_cfg)

    plot_channels = cfg.get("plot_channels", None)
    if plot_channels is not None:
        plot_channels = get_channels_from_strings(plot_channels)
    else:
        plot_channels = output_channels

    logger.info(f"Processing {len(times)} timestep(s): {times}")
    logger.info(f"Plot channels  : {get_strings_from_channels(plot_channels)}")
    logger.info(f"Input channels : {get_strings_from_channels(input_channels)}")
    logger.info(f"Output channels: {get_strings_from_channels(output_channels)}")
    
    input_channel_indices = []
    output_channel_indices = []
    for channel in plot_channels or []:
        input_channel_indices.append(input_channels.index(channel) if channel in input_channels else -1)
        output_channel_indices.append(output_channels.index(channel) if channel in output_channels else -1)

    output_path = Path(generation_dir) / cfg.get("results_dir_name", "evaluation_maps") / "snapshots"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    input_files = FileRepository(generation_dir)
    output_files = FileRepository(output_path)

    for curr_time in times:
        logger.info(f"Plotting timestep: {curr_time}")
        prediction = input_files.load(curr_time, f'{curr_time}-predictions')
        baseline = input_files.load(curr_time, f'{curr_time}-baseline')
        target = input_files.load(curr_time, f'{curr_time}-target')
        # When the target dataset has no data for this date it is saved as all zeros
        # (see AnemoiDataset.target_missing_as_zeros). With no ground truth available,
        # skip plotting the target and computing error metrics against it.
        target_missing = not np.any(target)
        if target_missing:
            logger.info(f"Target for {curr_time} is all zeros; skipping target plots and error metrics.")
        try:
            mean_pred = input_files.load(curr_time, f'{curr_time}-regression-prediction')
        except FileNotFoundError:
            mean_pred = None

        for idx, channel in enumerate(plot_channels):
            in_idx = input_channel_indices[idx]
            out_idx = output_channel_indices[idx]

            # TODO Implement that it plots just output or just input channel if the other is missing
            if in_idx == -1 or out_idx == -1:
                logger.warning(f"Channel {channel.name} not found in input or output channels. Skipping.")
                continue

            plot_title = f"{format_time_str(curr_time)}: {getattr(channel, 'title', channel.name if channel.level == '' else f'{channel.name}_{channel.level}')}"
            target_2d = target[out_idx, :, :]
            baseline_2d = baseline[in_idx, :, :]
            vmin, vmax = calculate_bounds(
                None if target_missing else target_2d,
                prediction[:, out_idx, :, :] if prediction.ndim > 3 else prediction[idx, :, :],
                None if channel.name == "tp" else baseline_2d,
                mean_pred[out_idx, :, :] if mean_pred is not None else None,
            )
            meta = ChannelMeta.get(channel, vmin=vmin, vmax=vmax)

            # Build sources to plot: (label, member_idx, 2D field).
            sources: list = []
            if not target_missing:
                sources.append(("target", None, target_2d))
            sources.append(("baseline", None, baseline_2d))
            if mean_pred is not None:
                sources.append(("mean-prediction", None, mean_pred[out_idx, :, :]))
            for m, p in _pred_members(prediction, out_idx):
                sources.append(("prediction", m, p))

            if channel.name == "tp":
                for label, m, data in sources:
                    save_field(label, data, meta, output_files, channel, curr_time,
                               member=m, plot_func=plot_map_precipitation,
                               title=plot_title, grid_cfg=grid_cfg)
                continue

            err_cmap = meta.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis'
            for label, m, data in sources:
                save_field(label, data, meta, output_files, channel, curr_time,
                           member=m, title=plot_title, grid_cfg=grid_cfg)
                if label == "target" or target_missing:
                    continue
                _, mae = compute_mae(data, target_2d)
                me = data - target_2d
                save_field(label, mae.reshape(data.shape), meta, output_files, channel, curr_time,
                           member=m, kind="mae", cmap=err_cmap, vmin=0, vmax=meta.err_vmax,
                           title=plot_title, grid_cfg=grid_cfg)
                save_field(label, me, meta, output_files, channel, curr_time,
                           member=m, kind="me", cmap=meta.me_cmap,
                           vmin=meta.err_vmin, vmax=meta.err_vmax,
                           title=plot_title, grid_cfg=grid_cfg)

        # Wind speed / direction (and combined wind+precip) plots
        wind_out = {ch.name: i for i, ch in enumerate(output_channels) if ch.name in ("10u", "10v")}
        wind_in  = {ch.name: i for i, ch in enumerate(input_channels)  if ch.name in ("10u", "10v")}
        if "10u" not in wind_out or "10v" not in wind_out:
            continue

        o_u, o_v = wind_out["10u"], wind_out["10v"]
        i_u, i_v = wind_in["10u"], wind_in["10v"]

        # Build wind sources: (label, member_idx, u_2d, v_2d).
        wind_sources: list = []
        if not target_missing:
            wind_sources.append(("target", None, target[o_u, :, :], target[o_v, :, :]))
        wind_sources.append(("baseline", None, baseline[i_u, :, :], baseline[i_v, :, :]))
        if mean_pred is not None:
            wind_sources.append(("mean-prediction", None, mean_pred[o_u, :, :], mean_pred[o_v, :, :]))
        for m, pu, pv in _pred_members(prediction, o_u, o_v):
            wind_sources.append(("prediction", m, pu, pv))

        title_speed = f"{format_time_str(curr_time)}: FF10m"
        title_dir   = f"{format_time_str(curr_time)}: DD10m"
        speed_meta  = ChannelMeta.get("10u", vmin=0, vmax=10)
        dir_meta    = ChannelMeta.get("10u", vmin=0, vmax=360)

        wind_kinds = (
            ("FF10m", speed_meta, "viridis", 0, 10,  "max",      title_speed),
            ("DD10m", dir_meta,   "twilight", 0, 360, "neither", title_dir),
        )
        for label, m, u, v in wind_sources:
            speed = np.hypot(u, v)
            direction = wind_direction(u, v)
            for kind, w_meta, w_cmap, w_vmin, w_vmax, w_extend, w_title in wind_kinds:
                data = speed if kind == "FF10m" else direction
                save_field(
                    f"{kind}-{label}", data, w_meta, output_files, None, curr_time,
                    member=m, cmap=w_cmap, vmin=w_vmin, vmax=w_vmax, extend=w_extend,
                    custom_path=output_files.wind_file(kind, curr_time, f"{kind}-{label}", m),
                    plot_func=plot_map, title=w_title, grid_cfg=grid_cfg,
                )

        # Combined wind-speed + precipitation maps
        tp_out_idx = next((i for i, ch in enumerate(output_channels) if ch.name == "tp"), None)
        if tp_out_idx is None:
            continue
        tp_in_idx = next((i for i, ch in enumerate(input_channels) if ch.name == "tp"), tp_out_idx)
        title_wp = f"{format_time_str(curr_time)}: FF10m + Precipitation"
        wp_dir = output_files._ensure_dir("wind_precip")

        wp_sources: list = []
        if not target_missing:
            wp_sources.append(("target", None, target[o_u, :, :], target[o_v, :, :], target[tp_out_idx, :, :]))
        wp_sources.append(("baseline", None, baseline[i_u, :, :], baseline[i_v, :, :], baseline[tp_in_idx, :, :]))
        if mean_pred is not None:
            wp_sources.append(("mean-prediction", None,
                               mean_pred[o_u, :, :], mean_pred[o_v, :, :], mean_pred[tp_out_idx, :, :]))
        for m, pu, pv, ptp in _pred_members(prediction, o_u, o_v, tp_out_idx):
            wp_sources.append(("prediction", m, pu, pv, ptp))

        for label, m, u, v, tp in wp_sources:
            suffix = f"{label}_{m:02d}" if m is not None else label
            plot_map_wind_precip(
                u, v, tp,
                str(wp_dir / f"{curr_time}-wind_precip-{suffix}"),
                title=title_wp, grid_cfg=grid_cfg,
            )

    logger.info(f"Snapshots saved to: {output_path}")

if __name__ == "__main__":
    main(parse_eval_cli(allow_times=True))
