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

    with open(os.path.join(input_directory), 'info', 'cosmo.yaml') as cosmo_file:
        cosmo_config = yaml.safe_load(cosmo_file)
    channels = cosmo_config['select']

    # Reshape predictions, if necessary
    # target is shape [channels, ensembles, points]
    # prediction is shape [channels, ensembles, x, y]
    prediction = prediction.reshape(*target.shape)

    latitudes = lat_lon[:,0]
    longitudes = lat_lon[:,1]
    
    # convert to torch
    target = torch.from_numpy(target)
    prediction = torch.from_numpy(prediction)

    errors = metrics.absolute_error(prediction[0,:,:], target[0,:,:])
    plotting.plot_error_projection(errors, latitudes, longitudes, os.path.join('plots/errors/', date))

if __name__ == "__main__":
    main()