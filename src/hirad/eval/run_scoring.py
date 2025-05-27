import os
import sys 

import metrics
import numpy as np
import plotting
import torch
import yaml

X = 352 # length of grid from N-S
Y = 544 # length of grid from E-W

def main():
    # TODO: Better arg parsing.
    if len(sys.argv) < 3:
          raise ValueError('Expected call run_scoring.py [input data directory] [predictions directory] [output plot directory]')
    
    input_directory = sys.argv[1]
    predictions_directory = sys.argv[2]
    output_directory = sys.argv[3]

    with open(os.path.join(input_directory, 'info', 'cosmo.yaml')) as cosmo_file:
        cosmo_config = yaml.safe_load(cosmo_file)
    target_channels = cosmo_config['select']

    with open(os.path.join(input_directory, 'info', 'era.yaml')) as era_file:
        era_config = yaml.safe_load(era_file)
    input_channels = era_config['select']

    lat_lon = torch.load(os.path.join(input_directory, 'info', 'cosmo-lat-lon'), weights_only=False)
    latitudes = lat_lon[:,0]
    longitudes = lat_lon[:,1]

    # Iterate over all files in the ground truth directory
    files = os.listdir(os.path.join(input_directory, 'cosmo'))
    files = sorted(files)


    # Plot power spectra
    # TODO: Handle ensembles
    prediction_tensor = np.ndarray([len(files), len(target_channels), X, Y])
    baseline_tensor = np.ndarray([len(files), len(input_channels), X, Y])
    target_tensor = np.ndarray([len(files), len(target_channels), X, Y])

    for i in range(len(files)):
        datetime = files[i]
        target = torch.load(os.path.join(input_directory, 'cosmo', datetime), weights_only=False)
        baseline = torch.load(os.path.join(input_directory, 'era-interpolated', datetime), weights_only=False)
        prediction = torch.load(os.path.join(predictions_directory, datetime), weights_only=False)

        # TODO: Handle ensembles
        prediction_1d = prediction.reshape(prediction.shape[0], X*Y)
        prediction_2d = prediction.reshape(prediction.shape[0], X, Y)
        
        baseline_1d = baseline.reshape(baseline.shape[0], X*Y)
        baseline_2d = baseline.reshape(baseline.shape[0], X, Y)
        
        target_1d = target.reshape(target.shape[0], X*Y)
        target_2d = target.reshape(target.shape[0], X, Y)

        baseline_tensor[i, :] = baseline_2d
        prediction_tensor[i, :] = prediction_2d
        target_tensor[i,:] = target_2d


    # Calc spectra
    for t_c in range(len(target_channels)):
        b_c = input_channels.index(target_channels[t_c])
        freqs = {}
        power = {}
        if b_c > -1:
            b_freq, b_power = metrics.average_power_spectrum(baseline_tensor[:,b_c,:,:].squeeze(), 2.0)
            freqs['baseline'] = b_freq
            power['baseline'] = b_power
            #plotting.plot_power_spectrum(b_freq, b_power, target_channels[t_c], os.path.join('plots/spectra/baseline2dt',  target_channels[t_c] + '-all_dates'))
        t_freq, t_power = metrics.average_power_spectrum(target_tensor[:,t_c,:,:].squeeze(), 2.0)
        freqs['target'] = t_freq
        power['target'] = t_power
        p_freq, p_power = metrics.average_power_spectrum(prediction_tensor[:,t_c,:,:].squeeze(), 2.0)
        # TODO: Uncomment when we have predictions
        #freqs['prediction'] = p_freq
        #power['prediction'] = p_power
        plotting.plot_power_spectra(freqs, power, target_channels[t_c], os.path.join(output_directory, 'spectra', target_channels[t_c] + '-alldates'))

    # store MAE as tensor of date:channel:ensembles:points
    # TODO:  Handle ensembles
    baseline_absolute_error = np.ndarray([len(files),len(target_channels),1,X*Y])
    prediction_absolute_error = np.ndarray([len(files),len(target_channels),1,X*Y])

    for i in range(len(files)):
        datetime = files[i]
        target = torch.load(os.path.join(input_directory, 'cosmo', datetime), weights_only=False)
        baseline = torch.load(os.path.join(input_directory, 'era-interpolated', datetime), weights_only=False)
        prediction = torch.load(os.path.join(predictions_directory, datetime), weights_only=False)


        prediction_1d = prediction.reshape(prediction.shape[0], 1, X*Y)
        prediction_2d = prediction.reshape(prediction.shape[0], 1, X, Y)

        # Get MAE
        for t_c in range(len(target_channels)):
            b_c = input_channels.index(target_channels[t_c])
            if b_c > -1:
                _, baseline_errors = metrics.compute_mae(baseline[b_c,:,:], target[t_c,:,:])
                baseline_absolute_error[i, t_c, :, :] = baseline_errors
            _, prediction_errors = metrics.compute_mae(prediction_1d[t_c,:,:], target[t_c,:,:])
            prediction_absolute_error[i, t_c, :, :] = prediction_errors
 

    print(f'baseline_absolute_error.shape={baseline_absolute_error.shape}, prediction_absolute_error.shape={prediction_absolute_error.shape}')
    # Average errors over ensembles
    baseline_mae = np.mean(baseline_absolute_error, axis=2)
    prediction_mae = np.mean(prediction_absolute_error, axis=2)
    
    # Average errors over time
    baseline_mae = np.mean(baseline_mae, axis=0)
    prediction_mae = np.mean(prediction_mae, axis = 0)

    print(f'baseline mean error = {np.mean(baseline_mae, axis=-1)}')
    print(f'prediction mean error = {np.mean(prediction_mae, axis=-1)}')

    # Plot the mean error onto the grid.
    for t_c in range(len(target_channels)):
        plotting.plot_error_projection(baseline_mae[t_c,:], latitudes, longitudes, os.path.join(output_directory, 'baseline-error' + target_channels[t_c] + '-' + 'average_over_time'))
        plotting.plot_error_projection(prediction_mae[t_c,:], latitudes, longitudes, os.path.join(output_directory, 'prediction-error' + target_channels[t_c] + '-' + 'average_over_time'))


if __name__ == "__main__":
    main()