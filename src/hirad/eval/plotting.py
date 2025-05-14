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

def plot_power_spectrum(x, filename):
    fig = plt.figure()
    plt.psd(x)
    logging.info(f'plotting values to {filename}')
    plt.savefig(filename)
    plt.close('all')