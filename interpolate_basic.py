import sys
import datetime
import argparse
import yaml

import anemoi.datasets
from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
from scipy.interpolate import griddata

# Default paths for input data sets
ERA_CONFIG = 'era.yaml'
COSMO_CONFIG = 'cosmo.yaml'

# Temperature ranges to use for plot gradient
MIN_TEMP = 230 # -43.15 C
MAX_TEMP = 320 # 46.85 C

def _read_input(era_config_file: str, cosmo_config_file: str, bound_to_cosmo_area=True) -> tuple[Dataset, Dataset]:
    """
    Read both ERA and COSMO data, optionally bounding to the COSMO data area, and return the 2m
    temperature values for the time range under COSMO.
    """
    # trim edge removes boundary
    with open(cosmo_config_file) as cosmo_file:
        cosmo_config = yaml.safe_load(cosmo_file)
    cosmo = open_dataset(cosmo_config)
    with open(era_config_file) as era_file:
        era_config = yaml.safe_load(era_file)
    era = open_dataset(era_config)
    # Subset the ERA dataset to have COSMO area/dates.
    start_date = cosmo.metadata()['start_date']
    end_date = cosmo.metadata()['end_date']
    # load era5 2m-temperature in the time-range of cosmo
    # area = N, W, S, E
    if bound_to_cosmo_area:
        min_lat_cosmo = min(cosmo.latitudes)
        max_lat_cosmo = max(cosmo.latitudes)
        min_lon_cosmo = min(cosmo.longitudes)
        max_lon_cosmo = max(cosmo.longitudes)
        era = open_dataset(era, start=start_date, end=end_date,
                        area=(max_lat_cosmo, min_lon_cosmo, min_lat_cosmo, max_lon_cosmo))
    else:
        era = open_dataset(era, start=start_date, end=end_date)
    print(cosmo.shape)
    print(cosmo.variables)
    print(era.shape)
    return (era, cosmo)

def _interpolate_basic(era: Dataset, cosmo: Dataset) -> np.ndarray[np.intp]:
    """Perform simple interpolation from ERA5 to COSMO grid for all data points in the COSMO date range."""
    grid = np.column_stack((era.longitudes, era.latitudes)) # stack lon-lat columns of era5 points
    # Check that our date ranges do in fact line up.
    assert (era.start_date == cosmo.start_date and 
            era.end_date == cosmo.end_date and 
            era.frequency == cosmo.frequency and
            era.shape[0] == cosmo.shape[0]), "ERA and COSMO date ranges or frequencies do not align."

    assert(set(cosmo.variables).issubset(era.variables)), "COSMO variables should be subset of ERA"
    
    interp_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes)) # stack lon-lat column of cosmo points
    interpolated_data = np.empty([10, cosmo.shape[3]])
    #interpolated_data = np.empty([cosmo.shape[0], cosmo.shape[3]])  # TODO: Full dataset
    print(interpolated_data.shape)

    # TODO: Replace the for loop if possible.
    # Each 100 iterations takes 30s, so entire 7000 points would take 35 minutes (per channel).
    
    for i in range(10):
    #for i in range(era.shape[0]):
        values = np.array(era[i,0,0,:]) # get era grid 2m-temperature values on the given date-time
        interpolated_data[i,:] = griddata(grid,values,interp_grid,method='linear') # interpolate era5 to cosmo grid using scipy griddata linear

    return interpolated_data

def _save_interpolation(values: np.ndarray[np.intp], filename: str):
    """Output interpolated data to a given filename, in PyTorch tensor format."""
    torch_data = torch.from_numpy(values)
    # TODO: Separate file for each datetime -- all channels.
    # dataset / cosmo  -> 20200101-0000
    # dataset / era_interpolated -> 20200101-0000
    # OR - save back as an anemoi dataset -- ask francesco
    torch.save(torch_data, filename)
    

def _get_plot_indices(era: Dataset, cosmo: Dataset) -> np.ndarray[np.intp]:
    """
    Get indices of ERA5 data that is in the bounding rectangle of COSMO data.
    This is useful for plotting in the case where read_input(..., bound_to_cosmo_area=False) was used.
    In this case, one would then feed e.g. era.latitudes[indices] into _plot_projection.
    """
    min_lat_cosmo = min(cosmo.latitudes)
    max_lat_cosmo = max(cosmo.latitudes)
    min_lon_cosmo = min(cosmo.longitudes)
    max_lon_cosmo = max(cosmo.longitudes)
    box_lat = np.logical_and(era.latitudes>=min_lat_cosmo,era.latitudes<=max_lat_cosmo)
    box_lon = np.logical_and(era.longitudes>=min_lon_cosmo,era.longitudes<=max_lon_cosmo)
    indices = np.where(box_lon*box_lat)
    return indices

def _plot_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str, cmap=None, vmin = None, vmax = None):
    """Plot observed or interpolated data in a scatter plot."""
    # TODO: Refactor this somehow, it's not really generalizing well across variables.
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig(filename)
    plt.close('all')

def interpolate_and_save(infile_era: str, infile_cosmo: str, outfile_torch: str, outfile_plots_prefix: str = None, plot_times = [0]):
    """Read both ERA and COSMO data and perform basic interpolation. Save output into Pytorch format, and (optionally) plot
    ERA, COSMO, and interpolated data.

    Parameters:
    infile_era: str
        Local file path to ERA5 data
    infile_cosmo: str 
        Local file path to COSMO2 data
    outfile_torch: str
        Local file path to intended output file
    outfile_plots_prefix: str (Optional)
        Local file path to plots. If specified, plots will be saved as "{plotfilepath_prefix}-(era|cosmo|interpolated).jpg"

    Returns: 
    tuple[Dataset, Dataset]
        A tuple of ERA and COSMO 2m temperature data, in anemoi Dataset format, restricted to COSMO's date ranges
        (optionally the COSMO area as well).
    """
    era, cosmo = _read_input(ERA_CONFIG, COSMO_CONFIG, bound_to_cosmo_area=True)

    istart = datetime.datetime.now()
    interpolated = _interpolate_basic(era, cosmo)
    iend = datetime.datetime.now()
    print(f'interpolation took {iend-istart} seconds')

    if outfile_plots_prefix:
        for i in plot_times:
            # TODO: use actual dates in filenames instead of i
            # plot era original
            _plot_projection(era.longitudes, era.latitudes, era[i, 0, 0, :], outfile_plots_prefix + "temperature-2m-era-" + str(i) + ".jpg", cmap='seismic', vmin=MIN_TEMP, vmax = MIN_TEMP)
            _plot_projection(era.longitudes, era.latitudes, era[i, 1, 0, :], outfile_plots_prefix + "wind-u-era-" + str(i) + ".jpg")
            _plot_projection(era.longitudes, era.latitudes, era[i, 2, 0, :], outfile_plots_prefix + "wind-v-era-" + str(i) + ".jpg")
            _plot_projection(era.longitudes, era.latitudes, era[i, 3, 0, :], outfile_plots_prefix + "precip-era-" + str(i) + ".jpg")

            # plot cosmo original
            _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, 0, 0, :], outfile_plots_prefix + "temperaure-2m-cosmo-" + str(i) + ".jpg", cmap='seismic', vmin=MIN_TEMP, vmax = MIN_TEMP)
            _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, 1, 0, :], outfile_plots_prefix + "wind-u-cosmo-" + str(i) + ".jpg")
            _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, 2, 0, :], outfile_plots_prefix + "wind-v-cosmo-" + str(i) + ".jpg")
            _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, 3, 0, :], outfile_plots_prefix + "precip-cosmo-" + str(i) + ".jpg")

            #plot interpolated era5
            _plot_projection(cosmo.longitudes, cosmo.latitudes, interpolated[i,:], outfile_plots_prefix + "temperature-2m-interpolated-" + str(i) + ".jpg", cmap='seismic', vmin=MIN_TEMP, vmax = MIN_TEMP)

    #_save_interpolation(interpolated, outfile_torch)


def main():
    parser = argparse.ArgumentParser(
        prog='InterpolateBasic',
        description='Perform simple linear interpolation from ERA to COSMO grid'
    )
    parser.add_argument=('-')
    interpolate_and_save(ERA_CONFIG, COSMO_CONFIG, "interpolated.torch", "plots/")

if __name__ == "__main__":
    main()


