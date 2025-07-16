import logging
import os

from hirad.eval import crps, absolute_error

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_error_projection(values: np.array, latitudes: np.array, longitudes: np.array, filename: str, label: str, vmin=None, vmax=None):
    """Plot observed or interpolated data in a scatter plot."""
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    logging.info(f'plotting values to {filename}')
    p = ax.scatter(x=longitudes, y=latitudes, c=values, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label=label, orientation="horizontal")
    plt.savefig(filename)
    plt.close('all')

def _plot_score_vs_t(score: np.array, times: np.array, filename: str):
    fig = plt.figure()
    ax = plt.subplot()
    p = plt.plot(times, score)
    #plt.ylabel('CRPS')
    #plt.xlabel('time')
    plt.xticks([times[0],times[-1]])
    plt.savefig(filename)
    plt.close('all')

def plot_scores_vs_t(scores: dict[str,np.ndarray], times: np.array, filename: str):
    fig = plt.figure()
    ax = plt.subplot()
    colors = {'red', 'green', 'blue', 'orange'} # TODO, add more
    for k in scores.keys():
        p = ax.plot(times, scores[k], color=colors[k])
        p.set_label(k)
    ax.legend()
    #plt.ylabel('CRPS')
    #plt.xlabel('time')
    ax.set_xticks([times[0],times[-1]])
    plt.savefig(filename)
    plt.close('all')

def plot_power_spectra(freqs: dict, spec: dict, channel_name, filename):
    fig = plt.figure()
    for k in freqs.keys():
        plt.loglog(freqs[k], spec[k], label=k)
    plt.title(channel_name)
    plt.legend()
    plt.xlabel("Frequency (1/km)")
    plt.ylabel("Power Spectrum")
    plt.ylim(bottom=1e-1)
    #plt.psd(x)
    logging.info(f'plotting values to {filename}')
    plt.savefig(filename)
    plt.close('all')

def compute_crps_over_time(times, dataset, output_path):  
    logging.info('computing crps')  
    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()
    start_time=times[0]
    end_time=times[-1]

    # Load one prediction ensemble to get the shape
    prediction_ensemble = torch.load(os.path.join(output_path, times[0], f'{times[0]}-predictions'), weights_only=False)
    #all_predictions = np.ndarray((len(times), prediction_ensemble.shape[0], prediction_ensemble.shape[1], prediction_ensemble.shape[2], prediction_ensemble.shape[3]))
    #all_targets = np.ndarray((len(times), prediction_ensemble.shape[1], prediction_ensemble.shape[2], prediction_ensemble.shape[3]))
    
    for i in range(len(times)):
        logging.info(f'computing crps for {times[i]}')
        prediction_ensemble = torch.load(os.path.join(output_path, times[i], f'{times[i]}-predictions'), weights_only=False)
        baseline = torch.load(os.path.join(output_path, times[i], f'{times[i]}-baseline'), weights_only=False)
        target = torch.load(os.path.join(output_path, times[i], f'{times[i]}-target'), weights_only=False)

        # Calculate CRPS
        crps_area = crps(prediction_ensemble, target, average_over_area=False, average_over_channels=False)

        # Calculate ensemble mean error
        ensemble_mean = np.mean(prediction_ensemble, 0)
        ensemble_mean_error = absolute_error(ensemble_mean, target)

        # Calculate interpolation error (baseline #1)
        interpolation_error = np.zeros(baseline.shape)
        for j in range(len(output_channels)):
            index = -1
            for k in range(len(input_channels)):
                if input_channels[k].name == output_channels[j].name:
                    index = k
            if k > -1:
                interpolation_error[j,::] = baseline[k,::] - target[j,::]

        # Calculate persistence error (baseline #2)
        persistence_error = np.zeros(baseline.shape)
        if i > 0:
            prev = torch.load(os.path.join(output_path, times[i-1], f'{times[i-1]}-target'), weights_only=False)
            persistence_error = absolute_error(prev, target)
        else:
            persistence_error = absolute_error(target, target)
        

        torch.save(crps_area, os.path.join(output_path, times[i], f'{times[i]}-crps'))
        torch.save(ensemble_mean_error, os.path.join(output_path, times[i], f'{times[i]}-ensemble-mean-error'))
        torch.save(interpolation_error, os.path.join(output_path, times[i], f'{times[i]}-interpolation-error'))
        torch.save(persistence_error, os.path.join(output_path, times[i], f'{times[i]}-persistence-error'))

def compute_crps_over_time_and_area(times, dataset, output_path):
    logging.info('computing crps and errors')  
    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()
    start_time=times[0]
    end_time=times[-1]

    logging.info('calculating min/max')

    crps_area = torch.load(os.path.join(output_path, times[0], f'{times[0]}-crps'), weights_only=False)
    mins = np.ones((crps_area.shape[0])) * 999999999
    maxes = np.zeros((crps_area.shape[0]))

    for i in range(len(times)):
        if i % (24*5) == 0:
            logging.info(f'on time {times[i]}')
        # Shape = channels, x, y
        crps_area = torch.load(os.path.join(output_path, times[i], f'{times[i]}-crps'), weights_only=False)
        for j in range(crps_area.shape[0]):
            max = np.max(crps_area[j,::])
            min = np.min(crps_area[j,::])
            if max > maxes[j]:
                maxes[j] = max
            if min < mins[j]:
                mins[j] = min
    logging.info(f'maxes are {maxes}')
    logging.info(f'mins are {mins}')
    
    torch.save(maxes, os.path.join(output_path, f'crps-maxes-{times[0]}-{times[len(times)-1]}'))
    torch.save(mins, os.path.join(output_path, f'crps-mins-{times[0]}-{times[len(times)-1]}'))

    # make area and time plot
    total_crps_area = np.zeros((crps_area.shape[0],crps_area.shape[1],crps_area.shape[2]))
    total_ensemble_mean_area = np.zeros((crps_area.shape[0],crps_area.shape[1],crps_area.shape[2]))
    total_interpolation_area = np.zeros((crps_area.shape[0],crps_area.shape[1],crps_area.shape[2]))
    total_persistence_area = np.zeros((crps_area.shape[0],crps_area.shape[1],crps_area.shape[2]))

    crps_over_time = np.zeros((crps_area.shape[0], len(times)))
    ensemble_mean_over_time = np.zeros((crps_area.shape[0], len(times)))
    interpolation_over_time = np.zeros((crps_area.shape[0], len(times)))
    persistence_over_time = np.zeros((crps_area.shape[0], len(times)))
    for i in range(len(times)):
        if i % (24*5) == 0:
            logging.info(f'on time {times[i]}')
        crps_area = torch.load(os.path.join(output_path, times[i], f'{times[i]}-crps'), weights_only=False)
        total_crps_area = total_crps_area + crps_area
        ensemble_mean_area = torch.load(os.path.join(output_path, times[i], f'{times[i]}-ensemble-mean-error'), weights_only=False)
        total_ensemble_mean_area = total_ensemble_mean_area + ensemble_mean_area
        interpolation_area = torch.load(os.path.join(output_path, times[i], f'{times[i]}-interpolation-error'), weights_only=False)
        total_interpolation_area = total_interpolation_area + interpolation_area
        persistence_area = torch.load(os.path.join(output_path, times[i], f'{times[i]}-persistence-error'), weights_only=False)
        if i>0:
            total_persistence_area = total_persistence_area + persistence_area

        for j in range(crps_area.shape[0]):
            crps_over_time[j,i] = np.mean(crps_area[j,::])
            ensemble_mean_over_time[j,i] = np.mean(ensemble_mean_area[j,::])
            interpolation_over_time[j,i] = np.mean(interpolation_area[j,::])
            persistence_over_time[j,i] = np.mean(persistence_area[j,::])
    mean_crps_area = total_crps_area / len(times)
    mean_ensemble_mean_area = total_ensemble_mean_area / len(times)
    mean_interpolation_area = total_interpolation_area / len(times)
    mean_persistence_area = total_persistence_area / (len(times)-1)
    torch.save(mean_crps_area, os.path.join(output_path, f'crps-area-{times[0]}-{times[len(times)-1]}'))
    torch.save(mean_ensemble_mean_area, os.path.join(output_path, f'mae-ensemble-mean-area-{times[0]}-{times[len(times)-1]}'))
    torch.save(mean_interpolation_area, os.path.join(output_path, f'mae-interpolation-area-{times[0]}-{times[len(times)-1]}'))
    torch.save(mean_persistence_area, os.path.join(output_path, f'mae-persistence-area-{times[0]}-{times[len(times)-1]}'))


    torch.save(crps_over_time, os.path.join(output_path, f'crps-time-{times[0]}-{times[len(times)-1]}'))
    torch.save(ensemble_mean_over_time, os.path.join(output_path, f'mae-ensemble-mean-time-{times[0]}-{times[len(times)-1]}'))
    torch.save(interpolation_over_time, os.path.join(output_path, f'mae-interpolation-time-{times[0]}-{times[len(times)-1]}'))
    torch.save(persistence_over_time, os.path.join(output_path, f'mae-persistence-time-{times[0]}-{times[len(times)-1]}'))

def plot_crps_over_time_and_area(times, dataset, output_path):
    logging.info('plotting crps and errors')  
    longitudes = dataset.longitude()
    latitudes = dataset.latitude()
    input_channels = dataset.input_channels()
    output_channels = dataset.output_channels()
    start_time=times[0]
    end_time=times[-1]

    maxes = torch.load(os.path.join(output_path, f'crps-maxes-{start_time}-{end_time}'), weights_only=False)
    mins = torch.load(os.path.join(output_path, f'crps-mins-{start_time}-{end_time}'), weights_only=False)
    crps_time = torch.load(os.path.join(output_path, f'crps-time-{start_time}-{end_time}'), weights_only=False)
    crps_area = torch.load(os.path.join(output_path, f'crps-area-{start_time}-{end_time}'), weights_only=False)
    ensemble_mean_time = torch.load(os.path.join(output_path, f'mae-ensemble-mean-time-{start_time}-{end_time}'), weights_only=False)
    ensemble_mean_area = torch.load(os.path.join(output_path, f'mae-ensemble-mean-area-{start_time}-{end_time}'), weights_only=False)
    interpolation_time = torch.load(os.path.join(output_path, f'mae-interpolation-time-{start_time}-{end_time}'), weights_only=False)
    interpolation_area = torch.load(os.path.join(output_path, f'mae-interpolation-area-{start_time}-{end_time}'), weights_only=False)
    persistence_time = torch.load(os.path.join(output_path, f'mae-persistence-time-{start_time}-{end_time}'), weights_only=False)
    persistence_area = torch.load(os.path.join(output_path, f'mae-persistence-area-{start_time}-{end_time}'), weights_only=False)


    for j in range(crps_area.shape[0]):
        plot_error_projection(crps_area[j,::], latitudes, longitudes, os.path.join(output_path, f'NEW-crps-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                              label=output_channels[j].name)
        plot_error_projection(ensemble_mean_area[j,::], latitudes, longitudes, os.path.join(output_path, f'NEW-mae-ensemble-mean-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name)
        plot_error_projection(interpolation_area[j,::], latitudes, longitudes, os.path.join(output_path, f'NEW-mae-interpolation-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name)
        plot_error_projection(persistence_area[j,::], latitudes, longitudes, os.path.join(output_path, f'NEW-mae-persistence-area-{start_time}-{end_time}-{output_channels[j].name}.jpg'),
                        label=output_channels[j].name)
        
        _plot_score_vs_t(crps_time[j,::], times, os.path.join(output_path, f'NEW-crps-time-{start_time}-{end_time}-{output_channels[j].name}.jpg'))

        maes = {}
        maes['ensemble mean'] = ensemble_mean_time[j,::]
        maes['interpolation'] = interpolation_time[j,::]
        maes['persistence'] = persistence_time[j,::]
        plot_scores_vs_t(maes, times, os.path.join(output_path, f'NEW-mae-time-{start_time}-{end_time}-{output_channels[j].name}.jpg'))
        
