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

# Margin to use for ERA dataset (to avoid nans from interpolation at boundary)
ERA_MARGIN_DEGREES = 1.0

def read_anemoi_ds(config_file: str, start_date = None, end_date = None, area = None) -> Dataset:
    """Read an Anemoi dataset from config file, given (optional) date/area parameters.
    Start/end and area from config file will also be subsetted, if present,
    so start_date and end_date and area parameters will be additional subsetting,
    not an override.

    Parameters:
    config_file: str
        YAML file with anemoi recipe.
    start_date, end_date: str (optional)
        e.g. '2020-01-01', see anemoi open_dataset documentation.
    area: tuple
        (N, W, S, E) lat/lon lines to bound the area, see anemoi open_dataset.

    Returns: 
    Dataset
        anemoi.Dataset of the dataset in question
    """
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
    ds = open_dataset(config, start=start_date, end=end_date, area=area)
    return ds

def save_anemoi_latlon_grid(dataset: Dataset, filename: str):
    """Save lat/lon grid of an Anemoi dataset into a Torch file. (Note that
    array will have column 0 with latitudes, and column 1 with longitutdes)

    Parameters:
    dataset: anemoi.Dataset
        Dataset to extract lat/lon from.
    filename: str
        Full file path to output to.

    Returns: None
    """
    grid = np.column_stack((dataset.latitudes, dataset.longitudes))
    torch.save(grid, filename)

def save_anemoi_stats(dataset: Dataset, filename: str):
    """Save stats of an Anemoi dataset into a Torch file. (The torch file
    will be a dictionary of stat to value)

    Parameters:
    dataset: anemoi.Dataset
        Dataset to extract stats from.
    filename: str
        Full file path to output to.

    Returns: None
    """
    torch.save(dataset.statistics, filename)

def regrid(input_values_for_time: np.ndarray, input_grid: np.ndarray, output_grid: np.ndarray):
    """Regrid an array of values for a given time point from an input to output grid.

    Parameters:
    input_values_for_time: np.ndarray
        An array of dimension (channels, N) (where N = X x Y)
    filename: str
        Full file path to output to.

    Returns: None
    """
    # shape (channel, grid)
    assert(len(input_values_for_time.shape) == 2)
    interpolated_data = np.empty([input_values_for_time.shape[0], output_grid.shape[0]])
    for j in range(input_values_for_time.shape[0]):
        values = np.array(input_values_for_time[j,:]) # get era grid values on the given date-time and channel
        regrid = griddata(input_grid, values, output_grid, method='linear') # interpolate era5 to cosmo grid using scipy griddata linear
        interpolated_data[j,0,:] = regrid
    return interpolated_data

def format_date(dt64: np.datetime64) -> str:
    """Makes date string from date time point, for saving files."""
    return to_datetime(dt64).strftime('%Y%m%d-%H%M')

def _save_datetime_file(values: np.ndarray[np.intp], date: np.datetime64, filepath: str, format='torch'):
    """saves array of values for a given date into a torch file"""
    filename = os.path.join(filepath, format_date(date))
    if format == 'torch':
        torch.save(values, filename)
    elif format == 'numpy':
        np.save(filename, values)
    else:
        raise NotImplementedError(f'invalid format {format}; currently only ' \
        'output to torch or numpy')

def plot_projection(ax, longitudes: np.array, latitudes: np.array, values: np.array, cmap=None, vmin = None, vmax = None, s = None):
    """Plot observed or interpolated data in a scatter plot"""
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=s)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, orientation="horizontal")

def plot_and_save_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str, projection=ccrs.PlateCarree(), cmap=None, vmin = None, vmax = None, s = None):
    """Plot observed or interpolated data in a scatter plot and save to file."""
    # TODO: Refactor this somehow, it's not really generalizing well across variables.
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    logging.info(f'plotting values to {filename}')
    plot_projection(ax, longitudes, latitudes, values, cmap, vmin, vmax, s)
    plt.savefig(filename)
    plt.close('all')

def interpolate_anemoi_time_point_to_grid(i: int, ds: Dataset, ds_name: str, input_grid: np.ndarray, output_grid: np.ndarray, output_data_path: str, format='torch', output_plots_path: str = None, plot_indices=[0]):
    """Interpolate a given time index in a dataset from its grid (input_grid) 
    to an output_grid, and save the interpolated values. In certain cases,
    additionally save a plot of the input and interpolated data.
    
    
    i: int
        Index of time to interpolate (0 is the first time point). This will 
        correspond to ds.dates[i]
    ds: anemoi.Dataset
        anemoi.Dataset to interpolate
    ds_name: str
        Name for the dataset (e.g. 'era'). This name will be used for the output
        directory ('era-interpolated') and in the plot filenames.
    input_grid: np.ndarray
        An ndarray of shape (N,2), where N is the number of datapoints (N = X x Y).
        ATTN: Longitudes are column 0 and Latitudes are column 1.
        This should be equal to np.column_stack((ds.longitudes, ds.latitudes))
    output_grid: np.ndarray
        Target grid to interpolate to.
        An ndarray of shape (N,2), where N is the number of datapoints (N = X x Y).
        ATTN: Longitudes are column 0 and Latitudes are column 1.
    output_data_path: str
        Path of directory for output data.
    format: str (Optional)
        Format of output (torch or numpy)
    output_plots_path: str (Optional)
        Path of directory to output plots. If None, no plots will be created.
    plot_indices: Array (Optional)
        Indices of time points for which to plot data
    """
    logging.info('interpolating time point ' + format_date(ds.dates[i]))
    # remove ensemble (3rd) dimension
    interpolated_data = regrid(ds[i,:,0,:], input_grid=input_grid, output_grid=output_grid)
    logging.info(f'writing time point { format_date(ds.dates[i])} to files in path {output_data_path}')
    _save_datetime_file(interpolated_data, ds.dates[i], os.path.join(output_data_path), format=format)
    if output_plots_path and i in plot_indices:
        datestr = format_date(ds.dates[i])
        logging.info(f'plotting {datestr} to {output_plots_path}')
        #for j in range(10):
        for j in range(min(10, len(ds.variables))):
            var = ds.variables[j]
            # plot era original
            plot_and_save_projection(input_grid[:,0], input_grid[:,1], ds[i, j, 0, :], f'{output_plots_path}/{ds.variables[j]}-{datestr}-{ds_name}.jpg')
            # plot interpolated
            plot_and_save_projection(output_grid[:,0], output_grid[:,1], interpolated_data[j, 0, :], f'{output_plots_path}/{ds.variables[j]}-{datestr}-{ds_name}-interpolated.jpg')

def save_anemoi_time_point(i: int, ds: Dataset, ds_name: str, data_output_path: str, plots_output_path: str = None, plot_indices=[0], format='torch'):
    """Save a time point of anemoi data (either input or target) directly into a given format.
    If the time point is in the """
    _save_datetime_file(ds[i,:,0,:], ds.dates[i], data_output_path, format)
    datestr = format_date(ds.dates[i])
    if plots_output_path and i in plot_indices:
        for j,var in enumerate(ds.variables):
            plot_and_save_projection(ds.longitudes, ds.latitudes, ds[i, j, 0, :], f'{plots_output_path}/{var}-{datestr}-{ds_name}.jpg')


### Main method 1: Interpolate ERA grid
def _interpolate_anemoi_to_grid(infile_anemoi: str, ds_name: str, output_grid: np.ndarray, output_path: str, format='torch', plot_indices=[0]):
    """Perform basic interpolation on an input dataset in anemoi format from 
    its native grid to a given output grid.
    Save output as intermediate datetime files in a given format (torch/numpy)
    Optionally plot interpolated data.

    Parameters:
    infile_anemoi: str
        Path to an anemoi recipe in YAML format.
    ds_name: str
        Name for the dataset (e.g. 'era'). This name will be used for the output
        directory ('era-interpolated') and in the plot filenames.
    output_grid: np.ndarray
        An ndarray of shape (N,2), where N is the number of datapoints (N = X x Y).
        ATTN: Longitudes are column 0 and Latitudes are column 1.
    output_path: str
        Path of parent directory for output. (sub-directories for plots, info,
        and interpolated data will be created if they do not already exist)
    format: str (Optional)
        Format of output (torch or numpy)
    plot_indices: Array (Optional)
        Indices of time points for which to plot data
    """

    os.makedirs(os.path.join(output_path, 'info'), exist_ok=True)
    os.makedirs(os.path.join(output_path, f'{ds_name}-interpolated'), exist_ok=True)

    # read data
    lats = output_grid[:,1]
    lons = output_grid[:,0]
    min_lat = min(lats) - ERA_MARGIN_DEGREES
    max_lat = max(lats) + ERA_MARGIN_DEGREES
    min_lon = min(lons) - ERA_MARGIN_DEGREES
    max_lon = max(lons) + ERA_MARGIN_DEGREES
    area=(max_lat, min_lon, min_lat, max_lon)
    logging.info(f'projecting onto era area {area}')
    ds = read_anemoi_ds(infile_anemoi, area = area)
    logging.info('Successfully read input')
    
    # Output stats and grid
    save_anemoi_stats(ds, os.path.join(output_path, f'info/{ds_name}-stats'))
    save_anemoi_latlon_grid(ds, os.path.join(output_path, f'info/{ds_name}-lat-lon'))

    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_anemoi, os.path.join(output_path, f'info/{ds_name}.yaml'))

    input_grid = np.column_stack((ds.longitudes, ds.latitudes))

    for i in range(len(ds.dates)):
        interpolate_anemoi_time_point_to_grid(i, ds, ds_name, input_grid, output_grid,
                                           os.path.join(output_path, f'{ds_name}-interpolated'),
                                           format=format,
                                           output_plots_path=os.path.join(output_path, 'plots'),
                                           plot_indices=plot_indices)


### Part 2: Save COSMO data
def _save_anemoi_as_format(infile_anemoi: str, ds_name: str, outfile_data_path: str, outfile_plots_path: str = None, plot_indices=[0], format='torch'):
    ds = read_anemoi_ds(infile_anemoi)
    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_anemoi, os.path.join(outfile_data_path, f'info/{ds_name}.yaml'))
    save_anemoi_stats(ds, os.path.join(outfile_data_path, f'info/{ds_name}-stats'))
    save_anemoi_latlon_grid(ds, os.path.join(outfile_data_path, f'info/{ds_name}-lat-lon'))
    ds_output_path = os.path.join(outfile_data_path, ds_name)
    os.makedirs(ds_output_path, exist_ok=True)
    for i in range(len(ds.dates)):
        save_anemoi_time_point(i, ds, ds_name, data_output_path=ds_output_path, outfile_plots_path=outfile_plots_path, plot_incides=[0], format=format)

    
def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call interpolate_basic.py [era.yaml] [cosmo.yaml] [output directory]')
    infile_era = sys.argv[1]
    infile_cosmo = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "info"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "era"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "cosmo"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "era-interpolated"), exist_ok=True)
    output_plots_path = os.path.join(output_path, "plots")
    os.makedirs(output_plots_path, exist_ok=True)

    erashortname = infile_era.split('/')[-1].split('.')[0]

    logging.basicConfig(
        filename=os.path.join(output_path, f'interpolate_basic-{erashortname}.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 

    logging.info(f'running {sys.argv}')
    #output_plots_path = None

    output_grid = None

    if infile_cosmo.endswith('yaml'):
        cosmo = read_anemoi_ds(infile_cosmo)
        output_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes))
    else:
        # This must be a lat-lon torch file.
        cosmo_latlon = torch.load(infile_cosmo, weights_only=False)
        lats = cosmo_latlon[:,0]
        lons = cosmo_latlon[:,1]
        output_grid = np.column_stack((lons, lats))

    _interpolate_era5_to_grid(infile_era, output_grid, output_path, plot_indices=[0])

if __name__ == "__main__":
    main()
