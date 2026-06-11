"""
Compute mean and std of Box-Cox-transformed channels from Anemoi zarr datasets.

Reads the same YAML config as AnemoiDataset (e.g. conf/dataset/anemoi_era_real.yaml)
and streams through every training timestep to compute single-pass statistics using
sum / sum-of-squares accumulation.

The printed output is formatted so it can be pasted directly into the YAML config
under transform_input_means / transform_input_stdevs / transform_output_means /
transform_output_stdevs.

Usage:
    python src/hirad/input_data/calculate_transformed_stats_anemoi.py \
        --config src/hirad/conf/dataset/anemoi_era_real.yaml \
        --output_dir ./transform_stats
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from anemoi.datasets import open_dataset
from tqdm import tqdm

from hirad.datasets.constants import ERA_TO_REAL_CHANNEL_MAP
from hirad.utils.dataset_utils import regrid_icon_to_rotlatlon, coarsen_2x

# Replicated from AnemoiDataset — margin around the target grid used when
# opening the ERA5 input dataset, so stats match what the model actually sees.
INPUT_MARGIN_DEGREES = 0.5

REAL_GRID_LAT_LON_PATH = "/capstor/store/cscs/pasc/c38/real_grid_info/realch1-lat-lon"


def parse_transform_descriptor(descriptor: str) -> tuple[str, str, float]:
    """'tp-box_cox_025' → ('tp', 'box_cox', 0.25)"""
    channel, *rest = descriptor.split("-")
    transformation = "-".join(rest)
    if transformation.startswith("box_cox"):
        lmbda_str = transformation.split("_")[-1]
        lmbda = float(lmbda_str) / (10 ** (len(lmbda_str) - 1))
        return channel, "box_cox", lmbda
    raise ValueError(f"Unknown transformation '{transformation}' in '{descriptor}'")


def box_cox_transform(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = np.clip(x, 0, None)
    return (np.power(x, lmbda) - 1) / lmbda


def apply_output_pipeline(
    values: np.ndarray,
    regrid_indices: torch.Tensor,
    regrid_weights: torch.Tensor,
    nx: int,
    ny: int,
    trim_edge: int,
    is_real2km_target: bool,
) -> np.ndarray:
    """Apply regrid → trim → (coarsen if real2km) to a flat array of grid points."""
    t = torch.from_numpy(values).unsqueeze(0)  # (1, n_points)
    t = regrid_icon_to_rotlatlon(t, regrid_indices, regrid_weights, nx=nx, ny=ny)
    if trim_edge > 0:
        t = t[:, trim_edge:-trim_edge, trim_edge:-trim_edge]
    if is_real2km_target:
        t = coarsen_2x(t)
    return t.numpy().ravel()


def accumulate_stats(dataset, channel_idx: int, transform, desc: str,
                     pipeline=None) -> tuple[float, float]:
    """Single-pass mean and std over all timesteps via sum / sum-of-squares.

    If pipeline is provided, it is applied to raw values before the transform.
    """
    total_sum = np.float64(0.0)
    total_sum_sq = np.float64(0.0)
    total_count = 0

    for t in tqdm(range(dataset.shape[0]), desc=desc, unit="step"):
        values = dataset[t].squeeze()[channel_idx].astype(np.float64).ravel()
        if pipeline is not None:
            values = pipeline(values)
        values = transform(values)
        total_sum += values.sum()
        total_sum_sq += (values ** 2).sum()
        total_count += len(values)

    mean = total_sum / total_count
    std = np.sqrt(max(total_sum_sq / total_count - mean ** 2, 0.0))
    return float(mean), float(std)


def derive_era5_area(cfg: dict) -> tuple:
    """Compute the ERA5 bounding box from the target grid, matching AnemoiDataset."""
    target_grid_path = cfg.get("target_grid_path", REAL_GRID_LAT_LON_PATH)
    target_dataset = cfg.get("type", "").split("_")[-1]
    is_real = target_dataset.startswith("real")
    if is_real:
        lat_lon = torch.load(target_grid_path, weights_only=False)
        latitudes = lat_lon[:, 0]
        longitudes = lat_lon[:, 1]
    else:
        # Open a single channel just to read latitudes/longitudes
        out_ch = (cfg.get("output_channel_names") or [])[0]
        _ds = open_dataset(cfg["target_anemoi_dataset_path"], select=[out_ch])
        latitudes = _ds.latitudes
        longitudes = _ds.longitudes

    min_lat = latitudes.min() - INPUT_MARGIN_DEGREES
    max_lat = latitudes.max() + INPUT_MARGIN_DEGREES
    min_lon = max(0.0, longitudes.min() - INPUT_MARGIN_DEGREES)
    max_lon = longitudes.max() + INPUT_MARGIN_DEGREES
    return (max_lat, min_lon, min_lat, max_lon)  # anemoi convention: N, W, S, E


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Box-Cox transform stats for Anemoi zarr datasets."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to dataset YAML config (e.g. conf/dataset/anemoi_era_real.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/transform_stats",
        help="Directory where per-channel stat files are saved (default: ./outputs/transform_stats).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = cfg["input_anemoi_dataset_path"]
    output_path = cfg["target_anemoi_dataset_path"]
    input_channel_names: list[str] = cfg["input_channel_names"]
    output_channel_names: list[str] = cfg["output_channel_names"]
    transform_channels: list[str] = cfg.get("transform_channels", [])
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    target_dataset = cfg.get("type", "").split("_")[-1]
    is_real_target = target_dataset.startswith("real")
    is_real2cosmo_target = target_dataset == "real2cosmo"
    is_real2km_target = target_dataset == "real2km"

    if not transform_channels:
        print("No transform_channels in config — nothing to do.")
        return

    output_pipeline = None
    if is_real_target:
        regrid_indices = torch.from_numpy(np.load(cfg["remap_indices_path"])).long()
        regrid_weights = torch.from_numpy(np.load(cfg["remap_weights_path"]))
        nx = 582 if is_real2cosmo_target else 1170
        ny = 390 if is_real2cosmo_target else 786
        trim_edge = cfg.get("trim_edge", 0)
        output_pipeline = lambda v: apply_output_pipeline(
            v, regrid_indices, regrid_weights, nx, ny, trim_edge, is_real2km_target
        )

    print("Deriving ERA5 area filter from target grid...")
    era5_area = derive_era5_area(cfg)
    print(f"  ERA5 area (N,W,S,E): {era5_area}")

    target_date_kwargs: dict = {}
    if start_date and end_date:
        target_date_kwargs["start"] = start_date
        target_date_kwargs["end"] = end_date

    results: dict[str, dict] = {}

    for descriptor in transform_channels:
        channel, transform_type, lmbda = parse_transform_descriptor(descriptor)
        print(f"\n=== {descriptor}  (channel={channel}, transform={transform_type}, lmbda={lmbda}) ===")
        transform = lambda x, _l=lmbda: box_cox_transform(x, _l)

        # --- Input (ERA5) ---
        in_mean = in_std = None
        if channel in input_channel_names:
            print(f"  Opening ERA5 input dataset for channel '{channel}'...")
            in_ds = open_dataset(
                input_path,
                select=[channel],
                start=start_date,
                end=end_date,
                area=era5_area,
            )
            print(f"  {in_ds.shape[0]} timesteps, {in_ds.shape[-1]} grid points")
            in_mean, in_std = accumulate_stats(in_ds, 0, transform, f"  input/{channel}")
            print(f"  Input  mean={in_mean:.10f}  std={in_std:.10f}")
            torch.save(in_mean, output_dir / f"input-{descriptor}-mean")
            torch.save(in_std, output_dir / f"input-{descriptor}-std")
        else:
            print(f"  '{channel}' not in input_channel_names — skipping input.")

        # --- Output (REAL-CH1 or COSMO) ---
        out_mean = out_std = None
        if channel in output_channel_names:
            real_ch = ERA_TO_REAL_CHANNEL_MAP[channel] if is_real_target else channel
            print(f"  Opening target output dataset for channel '{channel}' (select='{real_ch}')...")
            out_ds = open_dataset(output_path, select=[real_ch], **target_date_kwargs)
            print(f"  {out_ds.shape[0]} timesteps, {out_ds.shape[-1]} grid points")
            out_mean, out_std = accumulate_stats(out_ds, 0, transform, f"  output/{channel}", pipeline=output_pipeline)
            print(f"  Output mean={out_mean:.10f}  std={out_std:.10f}")
            torch.save(out_mean, output_dir / f"output-{descriptor}-mean")
            torch.save(out_std, output_dir / f"output-{descriptor}-std")
        else:
            print(f"  '{channel}' not in output_channel_names — skipping output.")

        results[descriptor] = dict(
            input_mean=in_mean, input_std=in_std,
            output_mean=out_mean, output_std=out_std,
        )

    # ---- Print yaml-paste-ready block ----
    print("\n\n" + "=" * 60)
    print("Paste into your dataset YAML config:")
    print("=" * 60)
    in_means  = {k: v["input_mean"]  for k, v in results.items() if v["input_mean"]  is not None}
    in_stds   = {k: v["input_std"]   for k, v in results.items() if v["input_std"]   is not None}
    out_means = {k: v["output_mean"] for k, v in results.items() if v["output_mean"] is not None}
    out_stds  = {k: v["output_std"]  for k, v in results.items() if v["output_std"]  is not None}
    print(f"transform_input_means:   {in_means}")
    print(f"transform_input_stdevs:  {in_stds}")
    print(f"transform_output_means:  {out_means}")
    print(f"transform_output_stdevs: {out_stds}")
    print("=" * 60)
    print(f"\nStat files saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()