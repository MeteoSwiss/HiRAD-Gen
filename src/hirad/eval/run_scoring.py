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
    #prediction_file = torch.load(os.path.join(predictions_directory, date), weights_only=False)
    prediction = torch.load(os.path.join(input_directory, 'cosmo', '20160101-0000'), weights_only=False)
    lat_lon = torch.load(os.path.join(input_directory, 'info', 'cosmo-lat-lon'), weights_only=False)

    # Reshape grides to be the same as prediction
    #target = target.squeeze().reshape(-1,*prediction.shape),
    target = torch.from_numpy(target)
    prediction = torch.from_numpy(prediction)
    #prediction = prediction.squeeze().reshape(-1,*prediction.shape)
    latitudes = lat_lon[:,0] #.squeeze().reshape(-1,*prediction.shape)
    longitudes = lat_lon[:,1] #squeeze().reshape(-1,*prediction.shape)
       
    errors = metrics.absolute_error(prediction[0,:,:], target[0,:,:])
    plotting.plot_error_projection(errors, latitudes, longitudes, os.path.join('plots/errors/', date))

if __name__ == "__main__":
    main()