import os
import sys 

import metrics
import plotting
import torch
import yaml


def main():
    if len(sys.argv) < 4:
          raise ValueError('Expected call run_scoring.py [input data directory] [predictions directory] [date]')
    
    input_directory = sys.argv[1]
    predictions_directory = sys.argv[2]
    date = sys.argv[3]

    target = torch.load(os.path.join(input_directory, 'cosmo', date), weights_only=False)
    baseline = torch.load(os.path.join(input_directory, 'era-interpolated', date), weights_only=False)
    prediction = torch.load(os.path.join(predictions_directory, date), weights_only=False)
    lat_lon = torch.load(os.path.join(input_directory, 'info', 'cosmo-lat-lon'), weights_only=False)

    with open(os.path.join(input_directory, 'info', 'cosmo.yaml')) as cosmo_file:
        cosmo_config = yaml.safe_load(cosmo_file)
    target_channels = cosmo_config['select']

    with open(os.path.join(input_directory, 'info', 'era.yaml')) as era_file:
        era_config = yaml.safe_load(era_file)
    input_channels = era_config['select']

    # Reshape predictions, if necessary
    # target is shape [channels, ensembles, points]
    # prediction is shape [channels, ensembles, x, y]
    prediction = prediction.reshape(*target.shape)

    latitudes = lat_lon[:,0]
    longitudes = lat_lon[:,1]
    
    # convert to torch
    target = torch.from_numpy(target)
    baseline = torch.from_numpy(baseline)
    prediction = torch.from_numpy(prediction)

    # plot errors
    for t_c in range(len(target_channels)):
        b_c = input_channels.index(target_channels[t_c])
        if b_c > -1:
            baseline_mae, baseline_errors = metrics.compute_mae(baseline[b_c,:,:], target[t_c,:,:])
            plotting.plot_error_projection(baseline_errors, latitudes, longitudes, os.path.join('plots/errors/', 'baseline', target_channels[t_c] + '-' + date))
        prediction_mae, prediction_errors = metrics.compute_mae(prediction[t_c,:,:], target[t_c,:,:])
        plotting.plot_error_projection(prediction_errors, latitudes, longitudes, os.path.join('plots/errors/', 'prediction', target_channels[t_c] + '-' + date))
    print(f'baseline MAE={baseline_mae}, prediction MAE={prediction_mae}')

if __name__ == "__main__":
    main()