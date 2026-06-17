import logging
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from hirad.eval.eval_utils import GridConfig, DEFAULT_GRID_CONFIG


def plot_map(values: np.array,
             filename: str,
             label='',
             title='',
             vmin=None,
             vmax=None,
             cmap=None,
             extend='neither',
             norm=None,
             ticks=None,
             grid_cfg = DEFAULT_GRID_CONFIG,
             patch_idx=0,
             patch_size=None
             ):
    """Plot observed or interpolated data in a scatter plot."""
    logging.info(f'Creating map: {filename}')

    if patch_size is None:
        patch_size = (grid_cfg.height, grid_cfg.width)
    # TODO: implement properly plotting of pathces for patched diffusion inference inspection
    # n_col_stacked = math.ceil(704/patch_size[0])
    # last_start = (n_col_stacked-1) * patch_size[0]
    # # print(n_col_stacked)
    # latitudes_start = last_start-(patch_idx%n_col_stacked)*patch_size[0]
    # # print(latitudes_start)
    # longitudes_start = (patch_idx//n_col_stacked)*patch_size[1]
    # # print(longitudes_start)
    # lat = LAT if lat is None else lat
    # lon = LON if lon is None else lon
    # latitudes  = lat[RELAX_ZONE : RELAX_ZONE + patch_size[0]] #LAT[RELAX_ZONE+latitudes_start:RELAX_ZONE+latitudes_start+patch_size[0]] #LAT[RELAX_ZONE : RELAX_ZONE + 352]
    # longitudes = lon[RELAX_ZONE : RELAX_ZONE + patch_size[1]] #LON[RELAX_ZONE+longitudes_start:RELAX_ZONE+longitudes_start+patch_size[1]] #LON[RELAX_ZONE : RELAX_ZONE + 544]
    latitudes  = grid_cfg.lat[grid_cfg.relax_zone : grid_cfg.relax_zone + grid_cfg.height]
    longitudes = grid_cfg.lon[grid_cfg.relax_zone : grid_cfg.relax_zone + grid_cfg.width]
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
        cbar.set_ticklabels([f'{tick:g}' for tick in ticks])

    plt.tight_layout()
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_symmetric_vmax(values: np.ndarray, percentile: float = 99.0, fallback: float = 1.0) -> float:
    """Return a robust symmetric colorbar half-range based on absolute values."""
    vmax = float(np.nanpercentile(np.abs(values), percentile))
    if vmax != vmax or vmax <= 0:
        return fallback
    return vmax


def plot_difference_map(
    values: np.ndarray,
    filename: str,
    title: str = '',
    label: str = 'Difference',
    grid_cfg: GridConfig = DEFAULT_GRID_CONFIG,
    cmap: str = 'RdBu_r',
    percentile: float = 99.0,
    fixed_vmax: Optional[float] = None,
):
    """Plot a difference map with symmetric diverging bounds around zero."""
    vmax = fixed_vmax if fixed_vmax is not None else compute_symmetric_vmax(values, percentile=percentile)
    plot_map(
        values,
        filename,
        title=title,
        label=label,
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap,
        extend='both',
        grid_cfg=grid_cfg,
    )

def plot_map_precipitation(values, filename, title='', threshold=0.01, rfac=1000.0, grid_cfg=DEFAULT_GRID_CONFIG):
    """Plot precipitation data with specific colormap and thresholds."""
    # Scale and mask values below threshold
    values = rfac * values # m/h --> mm/h
    values = np.ma.masked_where(values <= threshold, values)

    # Predefined colors and bounds specific for precipitation
    colors = ['none', 'powderblue', 'dodgerblue', 'mediumblue',
              'forestgreen', 'limegreen', 'lawngreen',
              'yellow', 'gold', 'darkorange', 'red',
              'darkviolet', 'violet', 'thistle']
    bounds = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=len(colors), clip=False)

    plot_map(
        values, filename,
        cmap=cmap,
        norm=norm,
        ticks=bounds,
        title=title,
        label='mm/h',
        extend='max',
        grid_cfg=grid_cfg,
    )

def plot_map_temperature(values, filename, title='', grid_cfg=DEFAULT_GRID_CONFIG):
    """Plot 2m temperature data with Meteoswiss-style colormap."""
    colors = [
        "#1A33CC", "#3366FF", "#4C97FF", "#4CA8FF", "#00CCFF",
        "#DEE699", "#A6D473", "#6BBF4D", "#33AB26", "#009900",
        "#33B300", "#66CC00", "#99E600", "#CCFF00", "#FFFF00",
        "#FFCC00", "#FF9900", "#FF6600", "#FF3300", "#FF0000",
        "#EB00EB", "#FF40FF", "#FF80FF", "#FFBFFF",
    ]
    bounds = [-9, -7, -5, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=len(colors), clip=False)

    plot_map(
        values, filename,
        cmap=cmap,
        norm=norm,
        ticks=bounds,
        title=title,
        label='Temperature [°C]',
        extend='both',
        grid_cfg=grid_cfg,
    )


def plot_map_wind_precip(
    u: np.ndarray,
    v: np.ndarray,
    tp: np.ndarray,
    filename: str,
    title: str = '',
    tp_threshold: float = 0.1,
    tp_rfac: float = 1000.0,
    wind_vmax: float = 15.0,
    grid_cfg: GridConfig = DEFAULT_GRID_CONFIG,
):
    """Plot surface windspeed as filled background with precipitation overlaid.

    Parameters
    ----------
    u, v      : wind component arrays (H, W), in m/s
    tp        : total precipitation array (H, W), in m/h (ERA5 units)
    filename  : output path without extension
    tp_threshold : minimum precipitation to show in mm/h (after rfac scaling)
    tp_rfac   : conversion factor applied to tp before plotting (default 1000 → m/h → mm/h)
    wind_vmax : upper end of the wind-speed colorbar [m/s]
    """
    logging.info(f'Creating wind+precip map: {filename}')

    wind_speed = np.hypot(u, v)

    precip = tp_rfac * tp
    precip_masked = np.ma.masked_where(precip <= tp_threshold, precip)

    precip_colors = ['powderblue', 'dodgerblue', 'mediumblue',
                     'forestgreen', 'limegreen', 'lawngreen',
                     'yellow', 'gold', 'darkorange', 'red']
    precip_bounds = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    precip_cmap = ListedColormap(precip_colors)
    precip_norm = BoundaryNorm(precip_bounds, ncolors=len(precip_colors), clip=False)

    latitudes  = grid_cfg.lat[grid_cfg.relax_zone : grid_cfg.relax_zone + grid_cfg.height]
    longitudes = grid_cfg.lon[grid_cfg.relax_zone : grid_cfg.relax_zone + grid_cfg.width]
    lon2d, lat2d = np.meshgrid(longitudes, latitudes)

    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.RotatedPole(pole_longitude=-170.0, pole_latitude=43.0)},
    )

    # Background: wind speed
    wind_mesh = ax.pcolormesh(
        lon2d, lat2d, wind_speed,
        cmap='inferno', shading='auto', vmin=0, vmax=wind_vmax,
    )

    # Overlay: precipitation (semi-transparent so wind field remains visible)
    precip_mesh = ax.pcolormesh(
        lon2d, lat2d, precip_masked,
        cmap=precip_cmap, norm=precip_norm, shading='auto', alpha=0.75,
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.gridlines(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    _ = fig.colorbar(
        wind_mesh, ax=ax, label='Wind Speed [m/s]',
        orientation='horizontal', shrink=0.7, pad=0.04, extend='max',
    )

    cbar_precip = fig.colorbar(
        precip_mesh, ax=ax, label='Precipitation [mm/h]',
        orientation='vertical', shrink=0.6, pad=0.02, extend='max',
    )
    cbar_precip.set_ticks(precip_bounds)
    cbar_precip.set_ticklabels([f'{b:g}' for b in precip_bounds])

    fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


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