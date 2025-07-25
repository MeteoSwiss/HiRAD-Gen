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

LOG_INTERVAL = 24

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
        if isinstance(data, tuple):  # (mean, std)
            mean, std = data
            line, = plt.plot(hours, mean, label=label)
            plt.fill_between(hours,
                             np.array(mean)-std,
                             np.array(mean)+std,
                             alpha=0.3,
                             color=line.get_color())
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
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting diurnal cycle computation for 2m temperature and windspeed")
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    logger.info(f"Loaded {len(times)} timesteps to process")

    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    logger.info("Dataset and sampler initialized")

    out_root = Path(cfg.generation.io.output_path or './outputs')
    load = lambda ts, fn: torch.load(out_root/ts/fn, weights_only=False)

    # Find channel indices
    out_ch = {c.name: i for i, c in enumerate(dataset.output_channels())}
    in_ch  = {c.name: i for i, c in enumerate(dataset.input_channels())}
    t2m_out = out_ch.get('2t', out_ch.get('t2m'))
    t2m_in = in_ch.get('2t', in_ch.get('t2m', t2m_out))
    u_out = out_ch.get('10u')
    v_out = out_ch.get('10v')
    u_in = in_ch.get('10u', u_out)
    v_in = in_ch.get('10v', v_out)
    logger.info(f"2T channel indices - output: {t2m_out}, input: {t2m_in}")
    logger.info(f"10U/10V channel indices - output: {u_out}/{v_out}, input: {u_in}/{v_in}")

    stats_temp = {mode: defaultdict(list) for mode in ['target','baseline','prediction']}
    stats_wind = {mode: defaultdict(list) for mode in ['target','baseline','prediction']}

    for idx, ts in enumerate(times, 1):
        hr = hour_of(ts)
        target = load(ts, f"{ts}-target")
        baseline = load(ts, f"{ts}-baseline")
        # 2m temperature
        stats_temp['target'][hr].append(target[t2m_out].mean())
        stats_temp['baseline'][hr].append(baseline[t2m_in].mean())
        # windspeed using np.hypot
        stats_wind['target'][hr].append(np.hypot(target[u_out], target[v_out]).mean())
        stats_wind['baseline'][hr].append(np.hypot(baseline[u_in], baseline[v_in]).mean())

        preds = load(ts, f"{ts}-predictions")
        for member in preds:
            stats_temp['prediction'][hr].append(member[t2m_out].mean())
            stats_wind['prediction'][hr].append(np.hypot(member[u_out], member[v_out]).mean())

        if idx % LOG_INTERVAL == 0 or idx == len(times):
            logger.info(f"Processed {idx}/{len(times)} timesteps ({ts})")

    # Compute hourly means
    mean_cycle_temp = {
        mode: [np.mean(stats_temp[mode][h]) for h in sorted(stats_temp[mode])]
        for mode in ['target','baseline']
    }
    mean_cycle_wind = {
        mode: [np.mean(stats_wind[mode][h]) for h in sorted(stats_wind[mode])]
        for mode in ['target','baseline']
    }
    logger.info("Computed hourly mean statistics")

    # Ensemble cycles (mean ± std)
    hrs, pred_mean_temp, pred_std_temp = compute_ensemble(stats_temp['prediction'])
    hrs_w, pred_mean_wind, pred_std_wind = compute_ensemble(stats_wind['prediction'])
    logger.info("Computed ensemble statistics")

    # Prepare cyclic series
    cycle = lambda x: x + [x[0]]
    hrs_c = hrs + [24]
    hrs_w_c = hrs_w + [24]
    temp_lines = [cycle(mean_cycle_temp['target']), cycle(mean_cycle_temp['baseline']), (cycle(pred_mean_temp), cycle(pred_std_temp))]
    wind_lines = [cycle(mean_cycle_wind['target']), cycle(mean_cycle_wind['baseline']), (cycle(pred_mean_wind), cycle(pred_std_wind))]

    # Plot
    plot_paths = []
    fn1 = out_root/'diurnal_cycle_2t.png'
    save_plot(hrs_c, temp_lines, ['COSMO-2','ERA5','CorrDiff ± Std(Members)'], '2m Temperature [K]',
              'Diurnal Cycle of 2m Temperature', fn1)
    plot_paths.append(fn1)

    fn2 = out_root/'diurnal_cycle_windspeed.png'
    save_plot(hrs_w_c, wind_lines, ['COSMO-2','ERA5','CorrDiff ± Std(Members)'], 'Windspeed [m/s]',
              'Diurnal Cycle of Windspeed', fn2)
    plot_paths.append(fn2)

    logger.info(f"Plots saved: {', '.join(str(p) for p in plot_paths)}")

if __name__ == '__main__':
    main()
