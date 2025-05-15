import logging

import numpy as np
import torch

from scipy.signal import periodogram


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
