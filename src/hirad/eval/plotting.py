import logging

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

def plot_error_projection(values: np.array, latitudes: np.array, longitudes: np.array, filename: str):
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    logging.info(f'plotting values to {filename}')
    p = ax.scatter(x=longitudes, y=latitudes, c=values)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="absolute error", orientation="horizontal")
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