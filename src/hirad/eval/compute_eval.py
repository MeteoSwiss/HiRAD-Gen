import hydra
import logging
import os
import json
from omegaconf import OmegaConf, DictConfig
import torch
import numpy as np
import contextlib
import datetime
from pandas import to_datetime

from hirad.distributed import DistributedManager
from hirad.utils.console import PythonLogger, RankZeroLoggingWrapper
from concurrent.futures import ThreadPoolExecutor

from hirad.eval import absolute_error, crps, plot_scores_vs_t, plot_error_projection
from hirad.models import EDMPrecondSuperResolution, UNet
from hirad.inference import Generator
from hirad.utils.inference_utils import save_images, save_results_as_torch
from hirad.utils.function_utils import get_time_from_range
from hirad.utils.checkpoint import load_checkpoint

from hirad.datasets import get_dataset_and_sampler_inference

from hirad.utils.train_helpers import set_patch_shape

@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig) -> None: 

    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # Initialize logger
    logger = PythonLogger("generate")  # General python logger

    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range, time_format="%Y%m%d-%H%M") 
    
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    dataset, sampler = get_dataset_and_sampler_inference(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time
    )
    output_path = getattr(cfg.generation.io, "output_path", "./outputs")

    compute_crps_per_time(times, dataset, output_path)
    compute_crps_over_time_and_area(times, output_path)
    plot_crps_over_time_and_area(times, dataset, output_path)

def _get_data_path(output_path, time=None, filename=None):
    if time:
        return os.path.join(output_path, time, filename)
    else:
        return os.path.join(output_path, filename)

def load_data(output_path, time=None, filename=None):
    return torch.load(_get_data_path(output_path, time, filename), weights_only=False)

def save_data(data, output_path, time=None, filename=None):
    path = _get_data_path(output_path, time, filename)
    torch.save(data, path)
    
def compute_crps_per_time(times, dataset, output_path):  
    logging.info('Computing CRPS for each time point')  
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()
    start_time=times[0]

    # Load one prediction ensemble to get the shape
    prediction_ensemble = torch.load(os.path.join(output_path, start_time, f'{start_time}-predictions'), weights_only=False)
    
    # Get a map of output to input channel, for building baseline errors
    output_to_input_channel_map = {}
    for j in range(len(output_channels)):
            index = -1
            for k in range(len(input_channels)):
                if input_channels[k].name == output_channels[j].name:
                    index = k
            output_to_input_channel_map[j] = index


    for i in range(len(times)):
        curr_time = times[i]
        if i % (24*5) == 0:
            logging.info(f'on time {curr_time}')
        prediction_ensemble = load_data(output_path, time=curr_time, filename=f'{curr_time}-predictions')
        baseline = load_data(output_path, time=curr_time, filename=f'{curr_time}-baseline')
        target = load_data(output_path, time=curr_time, filename=f'{curr_time}-target')

        # Calculate ensemble mean error
        ensemble_mean = np.mean(prediction_ensemble, 0)
        ensemble_mean_error = absolute_error(ensemble_mean, target)

        # Calculate interpolation error (baseline #1)
        interpolation_error = np.zeros(target.shape)
        for j in range(len(output_channels)):
            k = output_to_input_channel_map[j]
            if k > -1:
                interpolation_error[j,::] = absolute_error(baseline[k,::], target[j,::])

        # Calculate persistence error (baseline #2)
        persistence_error = np.zeros(target.shape)
        if i > 0:
            prev = load_data(output_path, time=times[i-1], filename=f'{times[i-1]}-target')
            persistence_error = absolute_error(prev, target)
        else:
            # for the first time point, persist the next-time-point target.
            # This is fiction but it keeps the plots from looking weird.
            prev = load_data(output_path, time=times[i+1], filename=f'{times[i+1]}-target')
            persistence_error = absolute_error(prev, target)

        
        # Calculate CRPS
        crps_diffusion_area = crps(prediction_ensemble, target, average_over_area=False, average_over_channels=False)

        save_data(crps_diffusion_area, output_path, time=curr_time, filename=f'{curr_time}-crps-ensemble')
        save_data(ensemble_mean_error, output_path, time=curr_time, filename=f'{curr_time}-ensemble-mean-error')
        save_data(interpolation_error, output_path, time=curr_time, filename=f'{curr_time}-interpolation-error')
        save_data(persistence_error, output_path, time=curr_time, filename=f'{curr_time}-persistence-error')

def compute_crps_over_time_and_area(times, output_path):
    logging.info('computing crps and errors')  
    start_time=times[0]
    end_time=times[-1]

    # shape = (channels, x, y)
    crps_area = load_data(output_path, time=start_time, filename=f'{start_time}-crps-ensemble')
    num_channels = crps_area.shape[0]

    # make area and time plot
    total_crps_area = np.zeros_like(crps_area)
    total_ensemble_mean_area = np.zeros_like(crps_area)
    total_interpolation_area = np.zeros_like(crps_area)
    total_persistence_area = np.zeros_like(crps_area)

    crps_over_time = np.zeros((num_channels, len(times)))
    ensemble_mean_over_time = np.zeros_like(crps_over_time)
    interpolation_over_time = np.zeros_like(crps_over_time)
    persistence_over_time = np.zeros_like(crps_over_time)
    for i in range(len(times)):
        curr_time = times[i]
        if i % (24*5) == 0:
            logging.info(f'on time {times[i]}')
        crps_area = load_data(output_path, time=curr_time, filename=f'{curr_time}-crps-ensemble')
        total_crps_area = total_crps_area + crps_area

        ensemble_mean_area = load_data(output_path, time=curr_time, filename=f'{curr_time}-ensemble-mean-error')
        total_ensemble_mean_area = total_ensemble_mean_area + ensemble_mean_area
        interpolation_area = load_data(output_path, time=curr_time, filename=f'{curr_time}-interpolation-error')
        total_interpolation_area = total_interpolation_area + interpolation_area
        persistence_area = load_data(output_path, time=curr_time, filename=f'{curr_time}-persistence-error')
        if i>0:
            total_persistence_area = total_persistence_area + persistence_area

        for j in range(num_channels):
            crps_over_time[j,i] = np.mean(crps_area[j,::])
            ensemble_mean_over_time[j,i] = np.mean(ensemble_mean_area[j,::])
            interpolation_over_time[j,i] = np.mean(interpolation_area[j,::])
            persistence_over_time[j,i] = np.mean(persistence_area[j,::])
    mean_crps_area = total_crps_area / len(times)
    mean_ensemble_mean_area = total_ensemble_mean_area / len(times)
    mean_interpolation_area = total_interpolation_area / len(times)
    mean_persistence_area = total_persistence_area / (len(times)-1)
    save_data(mean_crps_area, output_path, filename=f'crps-ensemble-area-{start_time}-{end_time}')
    save_data(mean_ensemble_mean_area, output_path, filename=f'mae-ensemble-mean-area-{start_time}-{end_time}')
    save_data(mean_interpolation_area, output_path, filename=f'mae-interpolation-area-{start_time}-{end_time}')
    save_data(mean_persistence_area, output_path, filename=f'mae-persistence-area-{start_time}-{end_time}')

    # Little hack to make the plots look nicer, without having to change dimensions.
    persistence_over_time[:,0] = persistence_over_time[:,1]

    save_data(crps_over_time, output_path, filename=f'crps-ensemble-time-{start_time}-{end_time}')
    save_data(ensemble_mean_over_time, output_path, filename=f'mae-ensemble-mean-time-{start_time}-{end_time}')
    save_data(interpolation_over_time, output_path, filename=f'mae-interpolation-time-{start_time}-{end_time}')
    save_data(persistence_over_time, output_path, filename=f'mae-persistence-time-{start_time}-{end_time}')

def plot_crps_over_time_and_area(times, dataset, output_path):
    logging.info('plotting crps and errors')  
    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    output_channels = dataset.output_channels()
    start_time=times[0]
    end_time=times[-1]

    crps_area = load_data(output_path, filename=f'crps-ensemble-area-{start_time}-{end_time}')
    ensemble_mean_area = load_data(output_path, filename=f'mae-ensemble-mean-area-{start_time}-{end_time}')
    interpolation_area = load_data(output_path, filename=f'mae-interpolation-area-{start_time}-{end_time}')
    persistence_area = load_data(output_path, filename=f'mae-persistence-area-{start_time}-{end_time}')

    crps_ensemble_time = load_data(output_path, filename=f'crps-ensemble-time-{start_time}-{end_time}')
    ensemble_mean_time = load_data(output_path, filename=f'mae-ensemble-mean-time-{start_time}-{end_time}')
    interpolation_time = load_data(output_path, filename=f'mae-interpolation-time-{start_time}-{end_time}')
    persistence_time = load_data(output_path, filename=f'mae-persistence-time-{start_time}-{end_time}')

    for j in range(len(output_channels)):
        plot_error_projection(crps_area[j,::], latitudes, longitudes, 
                              _get_data_path(output_path, filename=f'NEW-crps-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                              label=output_channels[j].name, title=f'Mean absolute error: CRPS: {output_channels[j].name}')
        plot_error_projection(ensemble_mean_area[j,::], latitudes, longitudes,
                              _get_data_path(output_path, filename=f'NEW-mae-ensemble-mean-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name, title=f'Mean absolute error: Ensemble mean: {output_channels[j].name}')
        plot_error_projection(interpolation_area[j,::], latitudes, longitudes,
                              _get_data_path(output_path, filename=f'NEW-mae-interpolation-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name, title=f'Mean absolute error: Interpolation: {output_channels[j].name}')
        plot_error_projection(persistence_area[j,::], latitudes, longitudes,
                              _get_data_path(output_path, filename=f'NEW-mae-persistence-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name, title=f'Mean absolute error: Persistence: {output_channels[j].name}')
    
        maes = {}
        maes['interpolation'] = interpolation_time[j,::]
        maes['ensemble mean'] = ensemble_mean_time[j,::]
        maes['crps'] = crps_ensemble_time[j,:] 
        maes['persistence'] = persistence_time[j,::]
        # TODO: consider casting times to datetime objects to avoid warnings.
        # However, this seems to be working OK, and a direct cast causes plotting errors
        plot_scores_vs_t(maes, times,
                         _get_data_path(output_path, filename=f'NEW-error-plot-time-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                         title=f'Mean absolute error: {output_channels[j].name}', xlabel='time', ylabel='MAE')
        

if __name__ == "__main__":
    main()