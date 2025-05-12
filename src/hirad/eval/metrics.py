import numpy as np
import torch


# set up MAE calculation to be run for each channel for a given date/time (for target COSMO, prediction, and ERA interpolated)

# input will be a 2D tensor of values with the COSMO lat/lon.

# Extracted from physicsnemo/examples/weather/regen/paper_figures/score_inference.py

def absolute_error(pred, target):
    return torch.abs(pred-target)

def compute_mae(pred, target):    
    # Exclude any target NaNs (not expected, but precautionary)
    mask = ~np.isnan(target)
    pred = pred[:, mask]
    target = target[mask]

    return torch.mean(absolute_error(pred, target))

