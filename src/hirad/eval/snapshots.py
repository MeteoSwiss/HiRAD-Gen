"""Generates maps of precipitation, temperature, and wind components/speed/direction."""
import logging
import argparse
import yaml
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.eval import compute_mae, plot_map
from hirad.eval.plotting import plot_map_precipitation, wind_direction, GridConfig, DEFAULT_GRID_CONFIG
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.inference_utils import calculate_bounds

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
        return torch.load(self.root / time / filename, weights_only=False)

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

    grid_cfg = GridConfig(
        lat = np.arange(cfg.get("lat_start"), cfg.get("lat_end") + cfg.get("lat_step"), cfg.get("lat_step")),
        lon = np.arange(cfg.get("lon_start"), cfg.get("lon_end") + cfg.get("lon_step"), cfg.get("lon_step")),
        height = cfg.get("height"),
        width = cfg.get("width"),
        relax_zone = cfg.get("relax_zone")
    )

    generation_dir = cfg.get("inference_output_dir", None)
    if generation_dir is None:
        logger.error("No inference_output_dir specified in config.")
        return
    
    if not Path(generation_dir).exists() or not Path(generation_dir).is_dir():
        logger.error(f"Inference output directory {generation_dir} does not exist or is not a directory.")
        return

    generation_config_path = Path(generation_dir) / ".hydra" / "config.yaml"
    if not generation_config_path.exists():
        logger.error(f"Generation config file {generation_config_path} does not exist.")
        return

    with open(generation_config_path, "r") as f:
        gen_cfg = yaml.safe_load(f)

    if cfg.get("times_range", None):
        times = get_time_from_range(cfg.get("times_range"), time_format="%Y%m%d-%H%M")
    elif cfg.get("times", None):
        times = cfg.get("times")
    elif gen_cfg.get("generation").get("times_range", None):
        times = get_time_from_range(gen_cfg.get("generation").get("times_range"), time_format="%Y%m%d-%H%M")
    elif gen_cfg.get("generation").get("times", None):
        times = gen_cfg.get("generation").get("times")
    else:
        logger.error("No times or times_range specified in config or generation config.")
        return
        
    dataset_cfg = gen_cfg.get("dataset")
    input_channels = get_channels_from_strings(dataset_cfg.get("input_channel_names", []))
    output_channels = get_channels_from_strings(dataset_cfg.get("output_channels_names", []))
    if not input_channels or not output_channels:
        dataset_type = dataset_cfg.get("type")
        dataset = known_datasets[dataset_type](**dataset_cfg)
        input_channels = dataset.input_channels()
        output_channels = dataset.output_channels()

    plot_channels = cfg.get("plot_channels", None)
    if plot_channels is not None:
        plot_channels = get_channels_from_strings(plot_channels)
    else:
        plot_channels = output_channels

    logger.info(get_strings_from_channels(plot_channels))
    logger.info(get_strings_from_channels(input_channels))
    logger.info(get_strings_from_channels(output_channels))
    
    input_channel_indices = []
    output_channel_indices = []
    for channel in plot_channels or []:
        input_channel_indices.append(input_channels.index(channel) if channel in input_channels else -1)
        output_channel_indices.append(output_channels.index(channel) if channel in output_channels else -1)

    output_path = Path(generation_dir) / cfg.get("results_dir_name", "evaluation_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    input_files = FileRepository(generation_dir)
    output_files = FileRepository(output_path)

    for curr_time in times:
        prediction = input_files.load(curr_time, f'{curr_time}-predictions')
        baseline = input_files.load(curr_time, f'{curr_time}-baseline')
        target = input_files.load(curr_time, f'{curr_time}-target')
        try:
            mean_pred = input_files.load(curr_time, f'{curr_time}-regression-prediction')
        except:
            mean_pred = None
        
        for idx, channel in enumerate(plot_channels):
            # input_channel_idx = output_to_input_channel_map[idx]
            input_channel_idx = input_channel_indices[idx]
            output_channel_idx = output_channel_indices[idx]

            # TODO Implement that it plots just output or just input channel if the other is missing
            if input_channel_idx == -1 or output_channel_idx == -1:
                logger.warning(f"Channel {channel.name} not found in input or output channels. Skipping.")
                continue

            plot_title = f"{format_time_str(curr_time)}: {getattr(channel, 'title', channel.name if channel.level == '' else f'{channel.name}_{channel.level}')}"
            vmin, vmax = calculate_bounds(
                target[output_channel_idx,:,:],
                prediction[:,output_channel_idx,:,:] if prediction.ndim>3 else prediction[idx,:,:],
                baseline[input_channel_idx,:,:] if not channel.name == "tp" else None,
                mean_pred[output_channel_idx,:,:] if mean_pred is not None else None
            )
            metadata = ChannelMeta.get(channel, vmin=vmin, vmax=vmax)

            if channel.name == "tp":
                save_field(
                    "target", target[output_channel_idx, :, :], metadata, output_files, channel, curr_time,
                    plot_func=plot_map_precipitation, title=plot_title, grid_cfg=grid_cfg
                )
                save_field(
                    "baseline", baseline[input_channel_idx, :, :], metadata, output_files, channel, curr_time,
                    plot_func=plot_map_precipitation, title=plot_title, grid_cfg=grid_cfg
                )
                if prediction.ndim>3:
                    for member_idx in range(prediction.shape[0]):
                        save_field(
                            "prediction", prediction[member_idx, output_channel_idx, :, :], metadata, output_files, channel, curr_time,
                            member=member_idx, plot_func=plot_map_precipitation, title=plot_title, grid_cfg=grid_cfg
                        )
                else:
                    save_field(
                        "prediction", prediction[output_channel_idx, :, :], metadata, output_files, channel, curr_time,
                        plot_func=plot_map_precipitation, title=plot_title, grid_cfg=grid_cfg
                    )
                if mean_pred is not None:
                    save_field(
                        "mean-prediction", mean_pred[output_channel_idx, :, :], metadata, output_files, channel, curr_time,
                        plot_func=plot_map_precipitation, title=plot_title, grid_cfg=grid_cfg
                    )
                continue

            # Plot target and baseline and regression prediction if available
            save_field("target", target[output_channel_idx, :, :], metadata, output_files, channel, curr_time, title=plot_title, grid_cfg=grid_cfg)
            save_field("baseline", baseline[input_channel_idx, :, :], metadata, output_files, channel, curr_time, title=plot_title, grid_cfg=grid_cfg)
            if mean_pred is not None:
                save_field("mean-prediction", mean_pred[output_channel_idx, :, :], metadata, output_files, channel, curr_time, title=plot_title, grid_cfg=grid_cfg)

            # Baseline MAE and ME
            _, baseline_mae = compute_mae(baseline[input_channel_idx, :, :], target[output_channel_idx, :, :])
            baseline_me = (baseline[input_channel_idx, :, :] - target[output_channel_idx, :, :])
            save_field("baseline", baseline_mae.reshape(baseline[input_channel_idx, :, :].shape), metadata, output_files, channel, curr_time, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)
            save_field("baseline", baseline_me, metadata, output_files, channel, curr_time, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)

            # Regression prediction MAE and ME
            if mean_pred is not None:
                _, mean_mae = compute_mae(mean_pred[idx, :, :], target[output_channel_idx, :, :])
                mean_me = (mean_pred[output_channel_idx, :, :] - target[output_channel_idx, :, :])
                save_field("mean-prediction", mean_mae.reshape(mean_pred[output_channel_idx, :, :].shape), metadata, output_files, channel, curr_time, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)
                save_field("mean-prediction", mean_me, metadata, output_files, channel, curr_time, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)

            # Ensemble predictions
            if prediction.ndim > 3:
                for member_idx in range(prediction.shape[0]):
                    member = prediction[member_idx, output_channel_idx, :, :]
                    save_field("prediction", member, metadata, output_files, channel, curr_time, member=member_idx, title=plot_title, grid_cfg=grid_cfg)
                    _, prediction_mae = compute_mae(member, target[output_channel_idx, :, :])
                    save_field("prediction", prediction_mae.reshape(member.shape), metadata, output_files, channel, curr_time, member=member_idx, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)
                    prediction_me = (member - target[output_channel_idx, :, :])
                    save_field("prediction", prediction_me, metadata, output_files, channel, curr_time, member=member_idx, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)
            else:
                member = prediction[output_channel_idx, :, :]
                save_field("prediction", member, metadata, output_files, channel, curr_time, title=plot_title, grid_cfg=grid_cfg)
                _, prediction_mae = compute_mae(member, target[output_channel_idx, :, :])
                save_field("prediction", prediction_mae.reshape(member.shape), metadata, output_files, channel, curr_time, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)
                prediction_me = (member - target[output_channel_idx, :, :])
                save_field("prediction", prediction_me, metadata, output_files, channel, curr_time, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title, grid_cfg=grid_cfg)

        # Plot Windspeed and direction
        wind_channels = {ch.name: idx for idx, ch in enumerate(output_channels) if ch.name in ("10u", "10v")}
        wind_channels_input = {ch.name: idx for idx, ch in enumerate(input_channels) if ch.name in ("10u", "10v")}
        if "10u" in wind_channels and "10v" in wind_channels:
            idx_10u = wind_channels["10u"]
            idx_10v = wind_channels["10v"]
            input_idx_10u = wind_channels_input["10u"]
            input_idx_10v = wind_channels_input["10v"]

            # Compute windspeed and direction for target, baseline, prediction and mean prediction
            target_wind_speed = np.hypot(target[idx_10u, :, :], target[idx_10v, :, :])
            target_wind_dir = wind_direction(target[idx_10u, :, :], target[idx_10v, :, :])
            baseline_wind_speed = np.hypot(baseline[input_idx_10u, :, :], baseline[input_idx_10v, :, :])
            baseline_wind_dir = wind_direction(baseline[input_idx_10u, :, :], baseline[input_idx_10v, :, :])
            if prediction.ndim > 3:
                prediction_wind_speed = np.hypot(prediction[:, idx_10u, :, :], prediction[:, idx_10v, :, :])
                prediction_wind_dir = wind_direction(prediction[:, idx_10u, :, :], prediction[:, idx_10v, :, :])
            else:
                prediction_wind_speed = np.hypot(prediction[idx_10u, :, :], prediction[idx_10v, :, :])
                prediction_wind_dir = wind_direction(prediction[idx_10u, :, :], prediction[idx_10v, :, :])
            if mean_pred is not None:
                mean_wind_speed = np.hypot(mean_pred[idx_10u, :, :], mean_pred[idx_10v, :, :])
                mean_wind_dir = wind_direction(mean_pred[idx_10u, :, :], mean_pred[idx_10v, :, :])

            plot_title_speed = f"{format_time_str(curr_time)}: FF10m"
            plot_title_dir = f"{format_time_str(curr_time)}: DD10m"

            wind_meta = ChannelMeta.get("10u", vmin=0, vmax=10)
            dir_meta = ChannelMeta.get("10u", vmin=0, vmax=360)

            # Save windspeed plots
            save_field(
                "FF10m-target", target_wind_speed, wind_meta, output_files, None, curr_time,
                cmap="viridis", vmin=0, vmax=10, extend='max',
                custom_path=output_files.wind_file("FF10m", curr_time, "FF10m-target"),
                plot_func=plot_map, title=plot_title_speed, grid_cfg=grid_cfg
            )
            save_field(
                "FF10m-baseline", baseline_wind_speed, wind_meta, output_files, None, curr_time,
                cmap="viridis", vmin=0, vmax=10, extend='max',
                custom_path=output_files.wind_file("FF10m", curr_time, "FF10m-baseline"),
                plot_func=plot_map, title=plot_title_speed, grid_cfg=grid_cfg
            )
            if prediction.ndim > 3:
                for member_idx in range(prediction.shape[0]):
                    save_field(
                        "FF10m-prediction", prediction_wind_speed[member_idx], wind_meta, output_files, None, curr_time,
                        member=member_idx, cmap="viridis", vmin=0, vmax=10, extend='max',
                        custom_path=output_files.wind_file("FF10m", curr_time, "FF10m-prediction", member_idx),
                        plot_func=plot_map, title=plot_title_speed, grid_cfg=grid_cfg
                    )
            else:
                save_field(
                    "FF10m-prediction", prediction_wind_speed, wind_meta, output_files, None, curr_time,
                    cmap="viridis", vmin=0, vmax=10, extend='max',
                    custom_path=output_files.wind_file("FF10m", curr_time, "FF10m-prediction"),
                    plot_func=plot_map, title=plot_title_speed, grid_cfg=grid_cfg
                )
            if mean_pred is not None:
                save_field(
                    "FF10m-mean-prediction", mean_wind_speed, wind_meta, output_files, None, curr_time,
                    cmap="viridis", vmin=0, vmax=10, extend='max',
                    custom_path=output_files.wind_file("FF10m", curr_time, "FF10m-mean-prediction"),
                    plot_func=plot_map, title=plot_title_speed, grid_cfg=grid_cfg
                )

            # Save wind direction plots
            save_field(
                "DD10m-target", target_wind_dir, dir_meta, output_files, None, curr_time,
                cmap="twilight", vmin=0, vmax=360,
                custom_path=output_files.wind_file("DD10m", curr_time, "DD10m-target"),
                plot_func=plot_map, title=plot_title_dir, grid_cfg=grid_cfg
            )
            save_field(
                "DD10m-baseline", baseline_wind_dir, dir_meta, output_files, None, curr_time,
                cmap="twilight", vmin=0, vmax=360,
                custom_path=output_files.wind_file("DD10m", curr_time, "DD10m-baseline"),
                plot_func=plot_map, title=plot_title_dir, grid_cfg=grid_cfg
            )
            if prediction.ndim > 3:
                for member_idx in range(prediction.shape[0]):
                    save_field(
                        "DD10m-prediction", prediction_wind_dir[member_idx], dir_meta, output_files, None, curr_time,
                        member=member_idx, cmap="twilight", vmin=0, vmax=360,
                        custom_path=output_files.wind_file("DD10m", curr_time, "DD10m-prediction", member_idx),
                        plot_func=plot_map, title=plot_title_dir, grid_cfg=grid_cfg
                    )
            else:
                save_field(
                    "DD10m-prediction", prediction_wind_dir, dir_meta, output_files, None, curr_time,
                    cmap="twilight", vmin=0, vmax=360,
                    custom_path=output_files.wind_file("DD10m", curr_time, "DD10m-prediction"),
                    plot_func=plot_map, title=plot_title_dir, grid_cfg=grid_cfg
                )
            if mean_pred is not None:
                save_field(
                    "DD10m-mean-prediction", mean_wind_dir, dir_meta, output_files, None, curr_time,
                    cmap="twilight", vmin=0, vmax=360,
                    custom_path=output_files.wind_file("DD10m", curr_time, "DD10m-mean-prediction"),
                    plot_func=plot_map, title=plot_title_dir, grid_cfg=grid_cfg
                )

    logger.info("Image loading and plotting completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    parser.add_argument("--times", nargs="+", help="One or more timesteps to plot (format: YYYYMMDD-HHMM). Overrides times/times_range/times_ranges in the config.")
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    if args.times:
        for k in ("times", "times_range", "times_ranges"):
            cfg.pop(k, None)
        cfg["times"] = args.times

    main(cfg)
