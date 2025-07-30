import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.utils.function_utils import get_time_from_range

# Constants
CONV_FACTOR = 100    # Convert meters to mm/h
WET_THRESHOLD = 0.1  # Threshold for wet-hour in mm/h
LOG_INTERVAL = 24    # Log progress every N timesteps

def hour_of(dt: str, fmt: str = "%Y%m%d-%H%M") -> int:
    return datetime.strptime(dt, fmt).hour


def compute_ensemble(hourly_values):
    hours = sorted(hourly_values)
    means = [np.mean(hourly_values[h]) for h in hours]
    stds  = [np.std(hourly_values[h])  for h in hours]
    return hours, means, stds


def save_plot(hours, lines, labels, ylabel, title, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,4))
    for data, label in zip(lines, labels):
        if isinstance(data, tuple):
            mean, std = data
            line, = plt.plot(hours, mean, label=label)
            plt.fill_between(hours, np.maximum(np.array(mean)-std, 0), np.array(mean)+std, alpha=0.3, color=line.get_color())
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

    logger.info("Starting computations for diurnal cycle of precipitation amount and wet-hours")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Loaded {len(times)} timesteps to process")

    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    logger.info("Dataset and sampler initialized")

    out_root = Path(cfg.generation.io.output_path or './outputs')
    load = lambda ts, fn: torch.load(out_root/ts/fn, weights_only=False) * CONV_FACTOR

    # Find channel indices
    out_ch = {c.name: i for i, c in enumerate(dataset.output_channels())}
    in_ch  = {c.name: i for i, c in enumerate(dataset.input_channels())}
    tp_out = out_ch['tp']; tp_in = in_ch.get('tp', tp_out)
    logger.info(f"TP channel indices - output: {tp_out}, input: {tp_in}")

    # Prepare data structures
    stats = {mode: defaultdict(list) for mode in ['target','baseline','prediction']}
    wet_stats = {mode: defaultdict(list) for mode in stats}
    
    # Load land mask
    lsm_dat = np.load('/iopsstor/scratch/cscs/davidle/HiRAD-Gen/lsm.npy')
    lsm=np.flip(lsm_dat.reshape(352,544),0)


    # Collect data
    for idx, ts in enumerate(times, 1):
        hr = hour_of(ts)
        target = load(ts, f"{ts}-target")[tp_out]
        baseline = load(ts, f"{ts}-baseline")[tp_in]
        
        # Mask target, baseline, and preds where lsm < 0.5
        land_mask = lsm >= 0.5
        target = target * land_mask
        baseline = baseline * land_mask

        stats['target'][hr].append(target)
        stats['baseline'][hr].append(baseline)
        wet_stats['target'][hr].append((target > WET_THRESHOLD).mean())
        wet_stats['baseline'][hr].append((baseline > WET_THRESHOLD).mean())

        preds = load(ts, f"{ts}-predictions")[:, tp_out]
        preds = preds * land_mask
        for member in preds:
            stats['prediction'][hr].append(member.mean())
            wet_stats['prediction'][hr].append((member > WET_THRESHOLD).mean())

        if idx % LOG_INTERVAL == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute hourly means
    mean_cycle = {
        mode: [np.mean(stats[mode][h]) for h in sorted(stats[mode])]
        for mode in ['target','baseline']
    }
    wet_cycle = {
        mode: [np.mean(wet_stats[mode][h]) * 100. for h in sorted(wet_stats[mode])]
        for mode in ['target','baseline']
    }
    logger.info("Computed hourly mean and wet-cycle statistics")

    # Ensemble cycles (mean ± std)
    hrs, pred_mean, pred_std = compute_ensemble(stats['prediction'])
    _, wet_mean, wet_std = compute_ensemble(wet_stats['prediction'])
    # Multiply ensemble wet-hour statistics by 100 for percentage
    wet_mean = [v * 100. for v in wet_mean]
    wet_std = [v * 100. for v in wet_std]
    logger.info("Computed ensemble statistics")

    # Prepare cyclic series
    cycle = lambda x: x + [x[0]]
    hrs_c = hrs + [24]
    amount_lines = [cycle(mean_cycle['target']), cycle(mean_cycle['baseline']), (cycle(pred_mean), cycle(pred_std))]
    wet_lines = [cycle(wet_cycle['target']), cycle(wet_cycle['baseline']), (cycle(wet_mean), cycle(wet_std))]

    # Log the lines to be plotted (debug)
    # logger.info(f"amount_lines: {amount_lines}")
    # logger.info(f"wet_lines: {wet_lines}")

    # Plot
    plot_paths = []
    fn1 = out_root/'diurnal_cycle_precip_amount.png'
    save_plot(hrs_c, amount_lines, ['COSMO-2','ERA5','CorrDiff ± Std(Members)'], 'Rain Rate (mm/h)',
              'Diurnal Cycle of Precip Amount', fn1)
    plot_paths.append(fn1)

    fn2 = out_root/'diurnal_cycle_precip_wethours.png'
    save_plot(hrs_c, wet_lines, ['COSMO-2','ERA5','Pred Mean ± Std'], 'Wet-Hour Fraction [%]',
              'Diurnal Cycle of Wet-Hours (>0.1 mm/h)', fn2)
    plot_paths.append(fn2)

    logger.info(f"Plots saved: {', '.join(str(p) for p in plot_paths)}")

if __name__ == '__main__':
    main()
