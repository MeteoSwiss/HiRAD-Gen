import logging
from dataclasses import dataclass

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path
from datetime import datetime
from hirad.datasets import get_channels_from_strings, get_strings_from_channels



@dataclass
class GridConfig:
    lat: np.ndarray
    lon: np.ndarray
    height: int
    width: int
    relax_zone: int

DEFAULT_GRID_CONFIG = GridConfig(
    lat=np.arange(-4.42, 3.36 + 0.02, 0.02),
    lon=np.arange(-6.82, 4.80 + 0.02, 0.02),
    height=352,
    width=544,
    relax_zone=19
)

# Constants for data processing
CONV_FACTOR_HOURLY = 1000  # Convert precip of ERA5 from meters to mm/h
CONV_FACTOR = CONV_FACTOR_HOURLY * 24   # Convert precip of ERA5 from from meters to mm/day
WET_THRESHOLD = 0.1  # Threshold for wet-hour in mm/h
LOG_INTERVAL = 24    # Log progress every N timesteps

LAND_SEA_MASK_PATH = '/capstor/store/mch/msopr/hirad-gen/eval/lsm.npy'

def get_channel_indices(dataset, channels=None):
    """
    Get channel indices for input and output channels from dataset.
    
    Args:
        dataset: Dataset object with input_channels() and output_channels() methods
        channels: Optional list of channel names to look up. If None, returns all channel mappings.
        
    Returns:
        dict: Dictionary with 'input' and 'output' keys, each containing channel name -> index mapping
        
    Example:
        indices = get_channel_indices(dataset, ['tp', '2t', '10u', '10v'])
        tp_out = indices['output']['tp']
        tp_in = indices['input'].get('tp', tp_out)  # Fallback to output index if not in input
    """
    out_ch = {get_strings_from_channels(c): i for i, c in enumerate(dataset.output_channels())}
    in_ch = {get_strings_from_channels(c): i for i, c in enumerate(dataset.input_channels())}
    
    if channels is None:
        return {'input': in_ch, 'output': out_ch}
    
    # Filter to requested channels only
    filtered_out = {ch: out_ch[ch] for ch in channels if ch in out_ch}
    filtered_in = {ch: in_ch[ch] for ch in channels if ch in in_ch}
    
    return {'input': filtered_in, 'output': filtered_out}

def load_land_sea_mask(path=LAND_SEA_MASK_PATH, height=352, width=544):
    """Load and retrun a land-sea mask as xarray DataArray."""
    lsm_data = np.load(path).reshape(height, width)
    return xr.DataArray(
        np.where(lsm_data >= 0.5, 1.0, np.nan),
        dims=['lat', 'lon'],
        coords={"lat": np.arange(height), "lon": np.arange(width)}
    )

def concat_and_group_diurnal(list_of_da, is_member=False, scale=1.0):
    """Helper to concatenate DataArrays and compute diurnal statistics."""
    da = xr.concat(list_of_da, dim="time").groupby("time.hour")
    if is_member:
        timmean = da.mean(dim='time') * scale
        mean = timmean.mean(dim='member')
        std = da.std(dim='member').mean(dim='time') * scale
    else:
        mean = da.mean(dim='time') * scale
        std = None
    return mean, std


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