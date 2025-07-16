import logging
import os

from hirad.eval import crps, absolute_error

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_error_projection(values: np.array, latitudes: np.array, longitudes: np.array, filename: str, label: str, title='', vmin=None, vmax=None):
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

def plot_scores_vs_t(scores: dict[str,np.ndarray], times: np.array, filename: str, xlabel='', ylabel='', title=''):
    
    fig = plt.figure()
    ax = plt.subplot()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] # TODO, add more
    i=0
    for k in scores.keys():
        style = colors[i]
        # If more than 50 points, don't connect lines
        if len(times) > 50:
            style = style + '.'
        else:
            style = style + '-'
        p, = ax.plot(times, scores[k], style)
        i=i+1
        p.set_label(k)
    ax.legend()
    ax.set_xticks([times[0],times[-1]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
