import logging

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# COSMO‑2 GRID:
LAT = np.arange(-4.42, 3.36 + 0.02, 0.02)
LON = np.arange(-6.82, 4.80 + 0.02, 0.02)
RELAX_ZONE = 19 # Number of points dropped on each side (relaxation zone)

def plot_map(values: np.array,
             filename: str,
             label='',
             title='',
             vmin=None,
             vmax=None,
             cmap=None,
             extend='neither',
             norm=None,
             ticks=None):
    """Plot observed or interpolated data in a scatter plot."""
    logging.info(f'Creating map: {filename}')

    latitudes  = LAT[RELAX_ZONE : RELAX_ZONE + 352]
    longitudes = LON[RELAX_ZONE : RELAX_ZONE + 544]
    lon2d, lat2d = np.meshgrid(longitudes, latitudes)

    fig, ax = plt.subplots(
        figsize=(8, 6),
        subplot_kw={"projection": ccrs.RotatedPole(pole_longitude=-170.0,
                                                   pole_latitude=  43.0)}
    )
    contour = ax.pcolormesh(
        lon2d, lat2d, values,
        cmap=cmap, shading="auto",
        norm=norm if norm else None,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.gridlines(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(title)
    cbar = plt.colorbar(
        contour,
        label=label,
        orientation="horizontal",
        extend=extend,
        shrink=0.75,
        pad=0.02
    )
    if ticks is not None:
        cbar.set_ticks(ticks)

    plt.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_map_precipitation(values, filename, title='', threshold=0.1, rfac=100.0):
    """Plot precipitation data with specific colormap and thresholds."""
    # Scale and mask values below threshold
    values = rfac * values # m/h --> mm/h
    values = np.ma.masked_where(values <= threshold, values)

    # Predefined colors and bounds specific for precipitation
    colors = ['none', 'powderblue', 'dodgerblue', 'mediumblue',
              'forestgreen', 'limegreen', 'lawngreen',
              'yellow', 'gold', 'darkorange', 'red',
              'darkviolet', 'violet', 'thistle']
    bounds = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 100, 150, 200]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=len(colors), clip=False)

    plot_map(
        values, filename,
        cmap=cmap,
        norm=norm,
        ticks=bounds,
        title=title,
        label='mm/h',
        extend='max'
    )

def wind_direction(u, v):
    """Compute wind direction from u and v components."""
    return(np.arctan2(-u, -v) * 180 / np.pi) % 360

@DeprecationWarning
def plot_error_projection(values: np.array, latitudes: np.array, longitudes: np.array, filename: str, label='', title='', vmin=None, vmax=None):
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