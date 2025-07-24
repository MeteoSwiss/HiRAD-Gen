"""Generates maps of precipitation, temperature, and wind components/speed/direction."""
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.eval import compute_mae, plot_map
from hirad.eval.plotting import plot_map_precipitation, wind_direction
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
    precip_kwargs: dict    = field(default_factory=lambda: {"threshold": 0.1, "rfac": 100.0})

    @classmethod
    def get(cls, ch_or_name: "ChannelMeta | str | None", *, vmin=None, vmax=None) -> "ChannelMeta":
        name = getattr(ch_or_name, "name", ch_or_name or "")
        base = CHANNELS.get(name) or cls(name=name)
        if vmin is not None or vmax is not None:
            return replace(base, vmin=vmin, vmax=vmax)
        return base

CHANNELS = {
    "tp": ChannelMeta(name="tp", cmap=None, unit="mm/h", extend="max", precip_kwargs={"threshold": 0.1, "rfac": 100.0}),
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

def map_output_to_input_channels(output_channels, input_channels):
    """
    Maps output channels to input channels based on their names.
    """
    return {
        j: next((k for k, input_channel in enumerate(input_channels) if input_channel.name == output_channel.name), -1)
        for j, output_channel in enumerate(output_channels)
    }

def save_field(name, data, meta, files, channel, t, member=None, kind=None, cmap=None, vmin=None, vmax=None, custom_path=None, plot_func=None, title=None, **plot_kwargs):
    """Save a field by plotting it with the appropriate function and parameters. Handles precipitation specially and supports custom paths."""
    # Determine output path
    suffix = f"{name}-{kind}" if kind else f"{name}"   
    out_path = custom_path or files.output_file(channel, t, suffix, member)

    # Choose plotting function
    plot = plot_func or plot_map

    # Case dependent plot parameters
    title = title or meta.name
    extend = 'max' if kind == 'mae' else meta.extend
    common = {'title': title, 'norm': meta.norm, 'extend': extend, **plot_kwargs}

    # Precipitation case
    if plot.__name__ == 'plot_map_precipitation':
        precip_args = {**meta.precip_kwargs, **{k: plot_kwargs[k] for k in meta.precip_kwargs if k in plot_kwargs}}
        plot(data, out_path, title=title, **precip_args)
    else:
        plot(data, out_path, vmin=vmin if vmin is not None else meta.vmin, vmax=vmax if vmax is not None else meta.vmax, cmap=cmap or meta.cmap, label=meta.unit, **common)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:

    # Initialize distributed manager
    DistributedManager.initialize()

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("plot_maps")

    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") 
        
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    has_lead_time = cfg.generation.get("has_lead_time", False)
    dataset, sampler = get_dataset_and_sampler_inference(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )

    output_path = getattr(cfg.generation.io, "output_path", "./outputs")
    files = FileRepository(output_path)

    for curr_time in times:
        prediction = files.load(curr_time, f'{curr_time}-predictions')
        baseline = files.load(curr_time, f'{curr_time}-baseline')
        target = files.load(curr_time, f'{curr_time}-target')
        
        input_channels = dataset.input_channels()
        output_channels = dataset.output_channels()
        output_to_input_channel_map = map_output_to_input_channels(output_channels, input_channels)

        for idx, channel in enumerate(output_channels):
            input_channel_idx = output_to_input_channel_map[idx]
            plot_title = f"{format_time_str(curr_time)}: {getattr(channel, 'title', channel.name)}"
            vmin, vmax = calculate_bounds(
                target[idx,:,:],
                prediction[:,idx,:,:],
                baseline[input_channel_idx,:,:]
            )
            metadata = ChannelMeta.get(channel, vmin=vmin, vmax=vmax)

            if channel.name == "tp":
                save_field(
                    "target", target[idx, :, :], metadata, files, channel, curr_time,
                    plot_func=plot_map_precipitation, title=plot_title
                )
                save_field(
                    "baseline", baseline[input_channel_idx, :, :], metadata, files, channel, curr_time,
                    plot_func=plot_map_precipitation, title=plot_title
                )
                if prediction.shape[0] > 1:
                    for member_idx in range(prediction.shape[0]):
                        save_field(
                            "prediction", prediction[member_idx, idx, :, :], metadata, files, channel, curr_time,
                            member=member_idx, plot_func=plot_map_precipitation, title=plot_title
                        )
                else:
                    save_field(
                        "prediction", prediction[0, idx, :, :], metadata, files, channel, curr_time,
                        plot_func=plot_map_precipitation, title=plot_title
                    )
                continue

            # Plot target and baseline
            save_field("target", target[idx, :, :], metadata, files, channel, curr_time, title=plot_title)
            save_field("baseline", baseline[input_channel_idx, :, :], metadata, files, channel, curr_time, title=plot_title)

            # Baseline MAE and ME
            _, baseline_mae = compute_mae(baseline[input_channel_idx, :, :], target[idx, :, :])
            baseline_me = (baseline[input_channel_idx, :, :] - target[idx, :, :])
            save_field("baseline", baseline_mae.reshape(baseline[input_channel_idx, :, :].shape), metadata, files, channel, curr_time, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title)
            save_field("baseline", baseline_me, metadata, files, channel, curr_time, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title)

            # Ensemble predictions
            for member_idx in range(prediction.shape[0]):
                member = prediction[member_idx, idx, :, :]
                save_field("prediction", member, metadata, files, channel, curr_time, member=member_idx, title=plot_title)
                _, prediction_mae = compute_mae(member, target[idx, :, :])
                save_field("prediction", prediction_mae.reshape(member.shape), metadata, files, channel, curr_time, member=member_idx, kind="mae", cmap=metadata.cmap if channel.name not in ("10u", "10v", "2t") else 'viridis', vmin=0, vmax=metadata.err_vmax, title=plot_title)
                prediction_me = (member - target[idx, :, :])
                save_field("prediction", prediction_me, metadata, files, channel, curr_time, member=member_idx, kind="me", cmap=metadata.me_cmap, vmin=metadata.err_vmin, vmax=metadata.err_vmax, title=plot_title)

        # Plot Windspeed and direction
        wind_channels = {ch.name: idx for idx, ch in enumerate(output_channels) if ch.name in ("10u", "10v")}
        if "10u" in wind_channels and "10v" in wind_channels:
            idx_10u = wind_channels["10u"]
            idx_10v = wind_channels["10v"]
            input_idx_10u = output_to_input_channel_map[idx_10u]
            input_idx_10v = output_to_input_channel_map[idx_10v]

            # Compute windspeed and direction for target, baseline, prediction
            target_wind_speed = np.hypot(target[idx_10u, :, :], target[idx_10v, :, :])
            target_wind_dir = wind_direction(target[idx_10u, :, :], target[idx_10v, :, :])
            baseline_wind_speed = np.hypot(baseline[input_idx_10u, :, :], baseline[input_idx_10v, :, :])
            baseline_wind_dir = wind_direction(baseline[input_idx_10u, :, :], baseline[input_idx_10v, :, :])
            prediction_wind_speed = []
            prediction_wind_dir = []
            for member_idx in range(prediction.shape[0]):
                ws = np.hypot(prediction[member_idx, idx_10u, :, :], prediction[member_idx, idx_10v, :, :])
                wd = wind_direction(prediction[member_idx, idx_10u, :, :], prediction[member_idx, idx_10v, :, :])
                prediction_wind_speed.append(ws)
                prediction_wind_dir.append(wd)
            prediction_wind_speed = np.stack(prediction_wind_speed)
            prediction_wind_dir = np.stack(prediction_wind_dir)

            plot_title_speed = f"{format_time_str(curr_time)}: FF10m"
            plot_title_dir = f"{format_time_str(curr_time)}: DD10m"

            wind_meta = ChannelMeta.get("10u", vmin=0, vmax=10)
            dir_meta = ChannelMeta.get("10u", vmin=0, vmax=360)

            # Save windspeed plots
            save_field(
                "FF10m-target", target_wind_speed, wind_meta, files, None, curr_time,
                cmap="viridis", vmin=0, vmax=10,
                custom_path=files.wind_file("FF10m", curr_time, "FF10m-target"),
                plot_func=plot_map, title=plot_title_speed
            )
            save_field(
                "FF10m-baseline", baseline_wind_speed, wind_meta, files, None, curr_time,
                cmap="viridis", vmin=0, vmax=10,
                custom_path=files.wind_file("FF10m", curr_time, "FF10m-baseline"),
                plot_func=plot_map, title=plot_title_speed
            )
            for member_idx in range(prediction.shape[0]):
                save_field(
                    "FF10m-prediction", prediction_wind_speed[member_idx], wind_meta, files, None, curr_time,
                    member=member_idx, cmap="viridis", vmin=0, vmax=10,
                    custom_path=files.wind_file("FF10m", curr_time, "FF10m-prediction", member_idx),
                    plot_func=plot_map, title=plot_title_speed
                )

            # Save wind direction plots
            save_field(
                "DD10m-target", target_wind_dir, dir_meta, files, None, curr_time,
                cmap="twilight", vmin=0, vmax=360,
                custom_path=files.wind_file("DD10m", curr_time, "DD10m-target"),
                plot_func=plot_map, title=plot_title_dir
            )
            save_field(
                "DD10m-baseline", baseline_wind_dir, dir_meta, files, None, curr_time,
                cmap="twilight", vmin=0, vmax=360,
                custom_path=files.wind_file("DD10m", curr_time, "DD10m-baseline"),
                plot_func=plot_map, title=plot_title_dir
            )
            for member_idx in range(prediction.shape[0]):
                save_field(
                    "DD10m-prediction", prediction_wind_dir[member_idx], dir_meta, files, None, curr_time,
                    member=member_idx, cmap="twilight", vmin=0, vmax=360,
                    custom_path=files.wind_file("DD10m", curr_time, "DD10m-prediction", member_idx),
                    plot_func=plot_map, title=plot_title_dir
                )

    logger.info("Image loading and plotting completed.")

if __name__ == "__main__":
    main()
