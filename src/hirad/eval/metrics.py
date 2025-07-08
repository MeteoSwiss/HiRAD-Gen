import logging

import numpy as np
import torch

from scipy.signal import periodogram
import xskillscore
import xarray as xr


# set up MAE calculation to be run for each channel for a given date/time (for target COSMO, prediction, and ERA interpolated)

# input will be a 2D tensor of values with the COSMO lat/lon.

# Extracted from physicsnemo/examples/weather/regen/paper_figures/score_inference.py

def absolute_error(pred, target) -> tuple[float, np.ndarray]:
    return np.abs(pred-target)

def compute_mae(pred, target):    
    # Exclude any target NaNs (not expected, but precautionary)
    # TODO: Fix the deprecated warning (index with dtype torch.bool instead of torch.uint8)
    mask = ~np.isnan(target)
    pred = pred[mask]
    target = target[mask]

    ae = absolute_error(pred, target)

    # TODO, consider adding axis=-1 to choose what axis to average
    return np.mean(absolute_error(pred, target)), ae

def average_power_spectrum(data: np.ndarray, d=2.0):  # d=2km by default
    """
    Compute the average power spectrum of a data array.

    This function calculates the power spectrum for each row of the input data and
    then averages them to obtain the overall power spectrum, repeating until
    dimensionality is reduced to 1D.
    The power spectrum represents the distribution of signal power as a function of frequency.

    Parameters:
        data (numpy.ndarray): Input data array.
        d (float): Sampling interval (time between data points).

    Returns:
        tuple: A tuple containing the frequency values and the average power spectrum.
        - freqs (numpy.ndarray): Frequency values corresponding to the power spectrum.
        - power_spectra (numpy.ndarray): Average power spectrum of the input data.
    """
    # Compute the power spectrum along the highest dimension for each row
    freqs, power_spectra = periodogram(data, fs=1 / d, axis=-1)
    logging.info(f'freqs.shape={freqs.shape}, power_spectra.shape={power_spectra.shape}')

    # Average along the first dimension
    while power_spectra.ndim > 1:
        power_spectra = power_spectra.mean(axis=0)
        logging.info(f'power spectra shape={power_spectra.shape}')

    return freqs, power_spectra

def crps(prediction_ensemble, target, average_over_area=True, average_over_channels=True, average_over_time=True):
    # Assumes that prediction_ensemble is in form:
    #  (member, channel, x, y) or
    #  (time, member, channel, x, y)
    # Returns: a k-dimensional array of continuous ranked probability scores,
    #   where k is the number of dimensions that were not averaged over.
    #   For example, if average_over_area is False (and all others true), will
    #   return an ndarray of shape (X,Y) 
    target_coords =  [('channel', np.arange(target.shape[-3])),
                        ('x', np.arange(target.shape[-2])),
                        ('y', np.arange(target.shape[-1]))]

    
    forecasts_coords = [('member', np.arange(prediction_ensemble.shape[-4])),
                        ('channel', np.arange(prediction_ensemble.shape[-3])),
                        ('x', np.arange(prediction_ensemble.shape[-2])),
                        ('y', np.arange(prediction_ensemble.shape[-1]))]
    
    if prediction_ensemble.ndim > 4 and target.ndim > 3:
        forecasts_coords.insert(0, ('time', np.arange(prediction_ensemble.shape[-5])))
        target_coords.insert(0, ('time', np.arange(target.shape[-4])))
        


    forecasts = xr.DataArray(prediction_ensemble, coords = forecasts_coords)
    observations = xr.DataArray(target, coords = target_coords)

    dim = []
    if prediction_ensemble.ndim > 4 and average_over_time:
        dim.append('time')
    if average_over_area:
        dim.append('x')
        dim.append('y')
    if average_over_channels:
        dim.append('channel')
    crps = xskillscore.crps_ensemble(observations=observations, forecasts=forecasts, dim=dim)
    crps = crps.to_numpy()
    return crps
