import logging
import netCDF4
from anemoi.datasets import open_dataset
import numpy as np
import yaml

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

import interpolate_basic

import sys
from pathlib import Path

import os
print (os.getcwd())

sys.path.insert(0, Path(__file__).parent.as_posix())

ANEMOI_1H_FILENAME = "/scratch/mch/omiralle/anemoi/aifs-ea-an-oper-0001-mars-n320-2015-2020-1h-v1-with-ERA51.zarr"
ANEMOI_6H_FILENAME = "/scratch/mch/apennino/data/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr"
COSMO_6H_FILENAME = "/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr"
COSMO_1H_FILENAME = "/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-1h-v3-pl13.zarr"
COSMO_CONFIG_FILE="src/input_data/cosmo.yaml"
CDF_FILENAME = "8e49f064d738154bed136666ff72ae1c.nc"


LAT = np.arange(-4.42, 3.36 + 0.02, 0.02)
LON = np.arange(-6.82, 4.80 + 0.02, 0.02)
RELAX_ZONE = 19 # Number of points dropped on each side (relaxation zone)


def extract_values(netcdf_data):
    netcdf_lat = netcdf_data['latitude'][:]
    netcdf_lon = netcdf_data['longitude'][:]
    netcdf_tp = netcdf_data['tp'][:,:]
    values = np.zeros((netcdf_tp.shape[0], netcdf_tp.shape[1]*netcdf_tp.shape[2]))
    latitudes = np.zeros(values.shape[1])
    longitudes = np.zeros(values.shape[1])
    # You could probably get this by reshaping, but I can't be bothered.
    for i in range(len(netcdf_lat)):
        if i % 10 == 0:
            print(i)
        for j in range(len(netcdf_lon)):
            grid_index = i * len(netcdf_lon) + j
            values[:,grid_index] = netcdf_tp[:,i,j]
            latitudes[grid_index] = netcdf_lat[i]
            longitudes[grid_index] = netcdf_lon[j]
    return values, latitudes, longitudes

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
    values = values.reshape((len(latitudes), len(longitudes)))
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

def plot_map_precipitation(values, filename, title='', threshold=0.1, rfac=1000.0):
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
        extend='max'
    )
 

print(interpolate_basic.regrid)
    
file_id = netCDF4.Dataset(CDF_FILENAME)
#anemoi1 = open_dataset(ANEMOI_1H_FILENAME)
#anemoi6 = open_dataset(ANEMOI_6H_FILENAME)
#with open(COSMO_CONFIG_FILE) as cosmo_file:
#    cosmo_config = yaml.safe_load(cosmo_file)
#cosmo = open_dataset(cosmo_config)
cosmo1 = open_dataset(COSMO_1H_FILENAME, trim_edge=19, select=['tp'],start='2016-01-01',end='2016-02-29')
cosmo6 = open_dataset(COSMO_6H_FILENAME, trim_edge=19, select=['tp'],start='2016-01-01',end='2016-02-29')


output_grid= np.column_stack((cosmo1.longitudes, cosmo1.latitudes))
print(output_grid.shape)
print(cosmo1[0,0,0,:].shape)

plot_map_precipitation(values=cosmo1[0,:], filename="cosmo1.png")
plot_map_precipitation(values=cosmo6[0,:], filename="cosmo6.png")

#fig = plt.figure()
#fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
#interpolate_basic.plot_projection(ax, longitudes=cosmo1.longitudes, latitudes=cosmo1.latitudes, values=cosmo1[0,:])
#fig.savefig('cosmo1.png')

#fig = plt.figure()
#fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
#interpolate_basic.plot_projection(ax, longitudes=cosmo1.longitudes, latitudes=cosmo1.latitudes, values=cosmo6[0,:])
#fig.savefig('cosmo6.png')


values, latitudes, longitudes = extract_values(netcdf_data=file_id)
input_grid=np.column_stack((longitudes, latitudes))
vals = values[0,:].reshape((1,1,values.shape[1]))
regrid=interpolate_basic.regrid(vals, input_grid, output_grid)
plot_map_precipitation(regrid, 'netcdf.png')


era1 = open_dataset(ANEMOI_1H_FILENAME, select=['tp'],start='2016-01-01',end='2016-02-29')
era6 = open_dataset(ANEMOI_6H_FILENAME, select=['tp'],start='2016-01-01',end='2016-02-29')
era_grid = np.column_stack((era1.longitudes, era1.latitudes))
era1_regrid = interpolate_basic.regrid(era1[0,:], era_grid, output_grid)
plot_map_precipitation(era1_regrid, "era1.png")



era6_regrid = interpolate_basic.regrid(era6[0,:], era_grid, output_grid)
plot_map_precipitation(era1_regrid, "era6.png")



