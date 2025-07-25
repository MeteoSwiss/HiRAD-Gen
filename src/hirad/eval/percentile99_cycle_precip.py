"""
Plots the diurnal cycle of the all-hour 99th percentile of
precipitation, a somewhat reliable measure of the precipitation intensity.

Each hour, member and type is treaded separately, to conserve memory... but if the 
period is long, this can still be a lot of data and thus an OOM error can occur.
"""
import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range

# Constants
CONV_FACTOR = 100    # Convert meters to mm/h
LOG_INTERVAL = 24    # Log progress every N timesteps


def hour_of(dt: str, fmt: str = "%Y%m%d-%H%M") -> int:
    return datetime.strptime(dt, fmt).hour


def save_plot(hours, lines, labels, ylabel, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    for data, label in zip(lines, labels):
        if isinstance(data, tuple):  # (mean, std)
            mean, std = data
            lower = np.maximum(np.array(mean) - std, 0)
            upper = np.array(mean) + std
            line, = plt.plot(hours, mean, label=label)
            plt.fill_between(hours, lower, upper, alpha=0.3, color=line.get_color())
        else:
            plt.plot(hours, data, label=label)
    plt.xlabel('Hour (UTC)')
    plt.xticks(range(0,25,3))
    plt.xlim(0,24)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    # Setup logging
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting 99th-percentile diurnal cycle computation")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Loaded {len(times)} timesteps to process")

    # Initialize dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    logger.info("Dataset and sampler initialized")

    # Output root and loader
    out_root = Path(cfg.generation.io.output_path or './outputs')
    load = lambda ts, fn: torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR

    # Find channel indices
    out_ch = {c.name: i for i, c in enumerate(dataset.output_channels())}
    in_ch  = {c.name: i for i, c in enumerate(dataset.input_channels())}
    tp_out = out_ch['tp']; tp_in = in_ch.get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Storage for diurnal cycles
    pct99_mean = {'target': [], 'baseline': [], 'prediction': []}
    pct99_std  = {'target': [], 'baseline': [], 'prediction': []}

    # -- Target and Baseline: compute per hour --
    for mode in ['target', 'baseline']:
        logger.info(f"Processing mode: {mode}")
        for h in list(range(24)):
            arrs = [
                load(ts, f"{ts}-{mode}")[tp_out if mode == 'target' else tp_in]
                for ts in times if hour_of(ts) == h
            ]
            stack = np.stack(arrs, axis=0)
            f99 = np.percentile(stack, 99, axis=0)
            pct99_mean[mode].append(f99.mean())
            pct99_std[mode].append(np.std(f99, axis=None))
            del arrs, stack, f99
            
    # -- Predictions: compute per hour per member, then mean+std across members --
    # Determine number of ensemble members
    sample = load(times[0], f"{times[0]}-predictions")  # [n_members, n_channels, lat, lon]
    data_sample = sample[:, tp_out]
    n_members = data_sample.shape[0]

    for h in list(range(24)):
        logger.info(f"Processing predictions for hour {h}")
        mem_f99 = []
        # for each ensemble member, gather its hourly fields
        for m in range(n_members):
            arrs = []
            for ts in times:
                if hour_of(ts) != h:
                    continue
                preds = load(ts, f"{ts}-predictions")  # [n_members, n_channels, ...]
                arrs.append(preds[m, tp_out])  # one field
            # stack over time and compute 99th percentile at each grid point
            stack_m = np.stack(arrs, axis=0)
            f99_m   = np.percentile(stack_m, 99, axis=0)
            mem_f99.append(f99_m.mean())
        # ensemble-level mean and std over member-wise percentiles
        pct99_mean['prediction'].append(np.mean(mem_f99))
        pct99_std['prediction'].append(np.std(mem_f99, axis=None))
        # clean up per-hour buffers
        del mem_f99, stack_m, f99_m

    # Prepare cyclic series
    cycle_fn = lambda x: x + [x[0]]
    hrs_c = list(range(24)) + [list(range(24))[0] + 24]
    pct99_lines = [
        cycle_fn(pct99_mean['target']),
        cycle_fn(pct99_mean['baseline']),
        (
            cycle_fn(pct99_mean['prediction']),
            cycle_fn(pct99_std['prediction'])
        )
    ]

    # Plot combined diurnal 99th-percentile cycle
    fn = out_root/'diurnal_cycle_precip_99th_percentile.png'
    save_plot(
        hrs_c,
        pct99_lines,
        ['COSMO-2','ERA5','CorrDiff 99th Pct ± Std'],
        'Rain Rate (mm/h)',
        'Diurnal Cycle of 99th-Percentile Precipitation',
        fn
    )
    logger.info(f"Combined plot saved: {fn}")

if __name__ == '__main__':
    main()
