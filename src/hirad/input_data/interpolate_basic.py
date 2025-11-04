import datetime
import logging
import os
import shutil
import sys
import yaml

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pandas import to_datetime
from scipy.interpolate import griddata
import torch
import multiprocessing
import xarray
from earthkit.geo.rotate import unrotate

# Margin to use for ERA dataset (to avoid nans from interpolation at boundary)
ERA_MARGIN_DEGREES = 1.0

def _read_era5_cosmo(era_config_file: str, cosmo_config_file: str) -> tuple[Dataset, Dataset]:
    """
    Read both ERA and COSMO data, optionally bounding to the COSMO data area, and return the 2m
    temperature values for the time range under COSMO.
    """
    # trim edge removes boundary
    cosmo = read_cosmo_anemoi(cosmo_config_file)
    # area = N, W, S, E
    min_lat = min(cosmo.latitudes) - ERA_MARGIN_DEGREES
    max_lat = max(cosmo.latitudes) + ERA_MARGIN_DEGREES
    min_lon = min(cosmo.longitudes) - ERA_MARGIN_DEGREES
    max_lon = max(cosmo.longitudes) + ERA_MARGIN_DEGREES
    start_date = cosmo.metadata()['start_date']
    end_date = cosmo.metadata()['end_date']
    era = read_era5_anemoi(era_config_file,
                           start_date = start_date, end_date = end_date,
                           area=(max_lat, min_lon, min_lat, max_lon))    
    return (era, cosmo)

def read_cosmo_anemoi(cosmo_config_file: str):
    with open(cosmo_config_file) as cosmo_file:
        cosmo_config = yaml.safe_load(cosmo_file)
    cosmo = open_dataset(cosmo_config)
    return cosmo

def read_era5_anemoi(era_config_file: str, start_date = None,
                    end_date = None, area=None):
    with open(era_config_file) as era_file:
        era_config = yaml.safe_load(era_file)
    era = open_dataset(era_config)
    era = open_dataset(era, start=start_date, end=end_date,
        area=area)
    return era

def regrid(era_for_time: np.ndarray, input_grid: np.ndarray, output_grid: np.ndarray):
    # shape (channel, ensemble, grid)
    interpolated_data = np.empty([era_for_time.shape[0], 1, output_grid.shape[0]])
    for j in range(era_for_time.shape[0]):
        values = np.array(era_for_time[j,0,:]) # get era grid values on the given date-time and channel
        regrid = griddata(input_grid, values, output_grid, method='linear') # interpolate era5 to cosmo grid using scipy griddata linear
        interpolated_data[j,0,:] = regrid
    return interpolated_data

def _interpolate_era5_cosmo_task(i: int, era: Dataset, cosmo: Dataset, input_grid: np.ndarray, output_grid: np.ndarray, intermediate_files_path: str, outfile_plots_path: str = None, plot_indices=[0]):
    logging.info('interpolating time point ' + _format_date(cosmo.dates[i]))
    interpolated_data = np.empty([era.shape[1], 1, cosmo.shape[3]])
    for j in range(era.shape[1]):
        values = np.array(era[i,j,0,:]) # get era grid values on the given date-time and channel
        regrid = griddata(input_grid, values, output_grid, method='linear') # interpolate era5 to cosmo grid using scipy griddata linear
        interpolated_data[j,0,:] = regrid
    logging.info(f'writing time point { _format_date(cosmo.dates[i])} to files in path {intermediate_files_path}')
    if (intermediate_files_path):
        _save_datetime_file(interpolated_data, era.variables, era.dates[i], os.path.join(intermediate_files_path, "era-interpolated/"))
        _save_datetime_file(era[i,:,:,:], era.variables, era.dates[i], os.path.join(intermediate_files_path, "era/"))
        _save_datetime_file(cosmo[i,:,:,:], cosmo.variables, cosmo.dates[i], os.path.join(intermediate_files_path, "cosmo/"))
    logging.info(f'finished writing time point { _format_date(cosmo.dates[i])}')

    if outfile_plots_path and i in plot_indices:
        datestr = _format_date(era.dates[i])
        logging.info(f'plotting {datestr} to {outfile_plots_path}')
        for j,var in enumerate(era.variables):
        # plot era original
            plot_and_save_projection(era.longitudes, era.latitudes, era[i, j, 0, :], f'{outfile_plots_path}{era.variables[j]}-{datestr}-era.jpg')

            plot_and_save_projection(cosmo.longitudes, cosmo.latitudes, interpolated_data[j, 0, :], f'{outfile_plots_path}{era.variables[j]}-{datestr}-era-interpolated.jpg')
        for j,var in enumerate(cosmo.variables):
            plot_and_save_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, j, 0, :], f'{outfile_plots_path}{cosmo.variables[j]}-{datestr}-cosmo.jpg')



def _interpolate_era5_cosmo_basic(era: Dataset, cosmo: Dataset, intermediate_files_path: str, threaded = True, outfile_plots_path: str =None, plot_indices=[0]):
    """Perform simple interpolation from ERA5 to COSMO grid for all data points in the COSMO date range.

    Parameters:
    era: Dataset
        Pre-loaded anemoi dataset for ERA
    cosmo: Dataset
        Pre-loaded anemoi dataset for COSMO
    intermediate_files_path
        If set, will save each date point to a new file.

    Returns: 
    np.ndarray
        4-D array of interpolated values. (date, variable, ensemble, grid-point)
    """
    # Check that our date ranges do in fact line up.
    assert (era.start_date == cosmo.start_date and 
            era.end_date == cosmo.end_date and 
            era.frequency == cosmo.frequency and
            era.shape[0] == cosmo.shape[0]), "ERA and COSMO date ranges or frequencies do not align."
    input_grid = np.column_stack((era.longitudes, era.latitudes)) # stack lon-lat columns of era5 points
    output_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes)) # stack lon-lat column of cosmo points
    
    dates = range(cosmo.shape[0])
    
    if (threaded):
        pool = multiprocessing.Pool()
        for i in dates:
            pool.apply_async(_interpolate_era5_cosmo_task, (i, era, cosmo, input_grid, output_grid, intermediate_files_path, outfile_plots_path, plot_indices))

        pool.close()
        pool.join()
    else:
        for i in dates:
            _interpolate_era5_cosmo_task(i, era, cosmo, input_grid, output_grid, intermediate_files_path, outfile_plots_path, plot_indices)

    return 

def _format_date(dt64: np.datetime64) -> str:
    """Makes date string from date time point, for saving files."""
    return to_datetime(dt64).strftime('%Y%m%d-%H%M')

def _save_datetime_file(values: np.ndarray[np.intp], variables: np.ndarray, date: np.datetime64, filepath: str):
    filename = filepath + _format_date(date)
    torch.save(values, filename)

def save_anemoi_latlon_grid(dataset: Dataset, filename: str):
    grid = np.column_stack((dataset.latitudes, dataset.longitudes))
    torch.save(grid, filename)

def save_anemoi_stats(dataset: Dataset, filename: str):
    torch.save(dataset.statistics, filename)

def plot_projection(ax, longitudes: np.array, latitudes: np.array, values: np.array, cmap=None, vmin = None, vmax = None, s = None):
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, orientation="horizontal")

def plot_and_save_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str, projection=ccrs.PlateCarree(), cmap=None, vmin = None, vmax = None, s = None):
    """Plot observed or interpolated data in a scatter plot."""
    # TODO: Refactor this somehow, it's not really generalizing well across variables.
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    logging.info(f'plotting values to {filename}')
    plot_projection(ax, longitudes, latitudes, values, cmap, vmin, vmax, s)
    plt.savefig(filename)
    plt.close('all')

def interpolate_era5_cosmo_and_save(infile_era: str, infile_cosmo: str, outfile_data_path: str, threaded=True, outfile_plots_path: str = None, plot_indices=[0]):
    """Read both ERA and COSMO data and perform basic interpolation. Save output into Pytorch format, and (optionally) plot
    ERA, COSMO, and interpolated data.

    Parameters:
    infile_era: str
        Local file path to ERA5 data
    infile_cosmo: str 
        Local file path to COSMO2 data
    outfile_data_path: str
        Local file path to intended output file
    outfile_plots_path: str (Optional)
        Local file path to plots. If specified, plots will be saved as "{plotfilepath_prefix}-(era|cosmo|interpolated).jpg"

    Returns: 
    tuple[Dataset, Dataset]
        A tuple of ERA and COSMO 2m temperature data, in anemoi Dataset format, restricted to COSMO's date ranges
        (optionally the COSMO area as well).
    """
    os.makedirs(outfile_data_path, exist_ok=True)
    os.makedirs(os.path.join(outfile_data_path, "info"), exist_ok=True)
    os.makedirs(os.path.join(outfile_data_path, "era"), exist_ok=True)
    os.makedirs(os.path.join(outfile_data_path, "cosmo"), exist_ok=True)
    os.makedirs(os.path.join(outfile_data_path, "era-interpolated"), exist_ok=True)

    if outfile_plots_path:  
        os.makedirs(outfile_plots_path, exist_ok=True)

    logging.info(f'reading input according to configs {infile_era} and {infile_cosmo}')
    era, cosmo = _read_input(infile_era, infile_cosmo, bound_to_cosmo_area=True)
    logging.info('Successfully read input')

    # Output stats and grid
    save_anemoi_stats(era, os.path.join(outfile_data_path, "info/era-stats"))
    save_anemoi_stats(cosmo, os.path.join(outfile_data_path, "info/cosmo-stats"))
    save_anemoi_latlon_grid(cosmo, os.path.join(outfile_data_path, "info/cosmo-lat-lon"))
    save_anemoi_latlon_grid(era, os.path.join(outfile_data_path, "info/era-lat-lon"))

    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_cosmo, os.path.join(outfile_data_path, "info/cosmo.yaml"))
    shutil.copy(infile_era, os.path.join(outfile_data_path, "info/era.yaml"))

    # generate interpolated data
    _interpolate_era5_cosmo_basic(era, cosmo, outfile_data_path, threaded=threaded, outfile_plots_path=outfile_plots_path, plot_indices=plot_indices)

    
def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call interpolate_basic.py [era.yaml] [cosmo.yaml] [output directory]')
    infile_era = sys.argv[1]
    infile_cosmo = sys.argv[2]
    output_directory = sys.argv[3]

    logging.basicConfig(
        filename=os.path.join(output_directory, 'interpolate_basic.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 
    
    interpolate_era5_cosmo_and_save(infile_era, infile_cosmo, output_directory, threaded=False, outfile_plots_path=None)

if __name__ == "__main__":
    main()
