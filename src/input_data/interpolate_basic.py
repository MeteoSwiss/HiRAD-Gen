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

# Number of workers in pool for threading
NUM_WORKERS=6

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
    logging.info(start_date)
    logging.info(end_date)
    # load era5 2m-temperature in the time-range of cosmo
    # area = N, W, S, E
    if bound_to_cosmo_area:
        min_lat = min(cosmo.latitudes) - ERA_MARGIN_DEGREES
        max_lat = max(cosmo.latitudes) + ERA_MARGIN_DEGREES
        min_lon = min(cosmo.longitudes) - ERA_MARGIN_DEGREES
        max_lon = max(cosmo.longitudes) + ERA_MARGIN_DEGREES
        era = open_dataset(era, start=start_date, end=end_date,
                        area=(max_lat, min_lon, min_lat, max_lon))
    else:
        era = open_dataset(era, start=start_date, end=end_date) 
    
    return (era, cosmo)

def _interpolate_task(i: int, era: Dataset, cosmo: Dataset, input_grid: np.ndarray, output_grid: np.ndarray, intermediate_files_path: str, outfile_plots_path: str = None, plot_indices=[0]):
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
            _plot_projection(era.longitudes, era.latitudes, era[i, j, 0, :], f'{outfile_plots_path}{era.variables[j]}-{datestr}-era.jpg')

            _plot_projection(cosmo.longitudes, cosmo.latitudes, interpolated_data[j, 0, :], f'{outfile_plots_path}{era.variables[j]}-{datestr}-era-interpolated.jpg')
        for j,var in enumerate(cosmo.variables):
            _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[i, j, 0, :], f'{outfile_plots_path}{cosmo.variables[j]}-{datestr}-cosmo.jpg')



def _interpolate_basic(era: Dataset, cosmo: Dataset, intermediate_files_path: str, threaded = True, outfile_plots_path: str =None, plot_indices=[0]):
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
        pool = multiprocessing.Pool(NUM_WORKERS)
        for i in dates:
            pool.apply_async(_interpolate_task, (i, era, cosmo, input_grid, output_grid, intermediate_files_path, outfile_plots_path, plot_indices))

        pool.close()
        pool.join()
    else:
        for i in dates:
            _interpolate_task(i, era, cosmo, input_grid, output_grid, intermediate_files_path, outfile_plots_path, plot_indices)

    return 

def _format_date(dt64: np.datetime64) -> str:
    """Makes date string from date time point, for saving files."""
    return to_datetime(dt64).strftime('%Y%m%d-%H%M')

def _save_datetime_file(values: np.ndarray[np.intp], variables: np.ndarray, date: np.datetime64, filepath: str):
    filename = filepath + _format_date(date)
    torch.save(values, filename)

def _save_latlon_grid(dataset: Dataset, filename: str):
    grid = np.column_stack((dataset.latitudes, dataset.longitudes))
    torch.save(grid, filename)

def _save_stats(dataset: Dataset, filename: str):
    torch.save(dataset.statistics, filename)

def _save_interpolation(values: np.ndarray[np.intp], filename: str):
    """Output interpolated data to a given filename, in PyTorch tensor format."""
    torch_data = torch.from_numpy(values)
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
    logging.info(f'plotting values to {filename}')
    p = ax.scatter(x=longitudes, y=latitudes, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig(filename)
    plt.close('all')

def interpolate_and_save(infile_era: str, infile_cosmo: str, outfile_data_path: str, threaded=True, outfile_plots_path: str = None, plot_indices=[0]):
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
    if not os.path.isdir(outfile_data_path):
        raise ValueError(f'Output path {outfile_data_path} does not exist.') 

    # Check for existence of subdirectories. TODO: Create them if missing.
    if not (os.path.isdir(os.path.join(outfile_data_path, "info")) and 
            os.path.isdir(os.path.join(outfile_data_path, "era")) and 
            os.path.isdir(os.path.join(outfile_data_path, "cosmo")) and 
            os.path.isdir(os.path.join(outfile_data_path, "era-interpolated"))):
        os.mkdir(os.path.join(outfile_data_path, "info"))
        os.mkdir(os.path.join(outfile_data_path, "era"))
        os.mkdir(os.path.join(outfile_data_path, "cosmo"))
        os.mkdir(os.path.join(outfile_data_path, "era-interpolated"))
    
    if outfile_plots_path and not os.path.isdir(outfile_plots_path):
        os.mkdir(outfile_plots_path)

    logging.info(f'reading input according to configs {infile_era} and {infile_cosmo}')
    era, cosmo = _read_input(infile_era, infile_cosmo, bound_to_cosmo_area=True)
    logging.info(f'Successfully read input')

    # Output stats and grid
    _save_stats(era, os.path.join(outfile_data_path, "info/era-stats"))
    _save_stats(cosmo, os.path.join(outfile_data_path, "info/cosmo-stats"))
    _save_latlon_grid(cosmo, os.path.join(outfile_data_path, "info/cosmo-lat-lon"))
    _save_latlon_grid(era, os.path.join(outfile_data_path, "info/era-lat-lon"))

    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_cosmo, os.path.join(outfile_data_path, "info/cosmo.yaml"))
    shutil.copy(infile_era, os.path.join(outfile_data_path, "info/era.yaml"))

    # generate interpolated data
    _interpolate_basic(era, cosmo, outfile_data_path, threaded=threaded, outfile_plots_path=outfile_plots_path, plot_indices=plot_indices)


def main():
    # TODO: Do better arg parsing so it's not as easy to reverse era and cosmo configs.
    if len(sys.argv) < 4:
        raise ValueError('Expected call interpolate_basic.py [era.yaml] [cosmo.yaml] [output directory]')
    infile_era = sys.argv[1]
    infile_cosmo = sys.argv[2]
    output_directory = sys.argv[3]


    logging.basicConfig(
        #filename='interpolate_basic.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S') 
    interpolate_and_save(infile_era, infile_cosmo, output_directory, threaded=False, outfile_plots_path=os.path.join(output_directory, "plots/"))

if __name__ == "__main__":
    main()
