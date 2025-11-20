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
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
    ds = open_dataset(config, start=start_date, end=end_date, area=area)
    return ds

def save_anemoi_latlon_grid(dataset: Dataset, filename: str):
    grid = np.column_stack((dataset.latitudes, dataset.longitudes))
    torch.save(grid, filename)

def save_anemoi_stats(dataset: Dataset, filename: str):
    torch.save(dataset.statistics, filename)

def regrid(era_for_time: np.ndarray, input_grid: np.ndarray, output_grid: np.ndarray):
    # shape (channel, ensemble, grid)
    interpolated_data = np.empty([era_for_time.shape[0], 1, output_grid.shape[0]])
    for j in range(era_for_time.shape[0]):
        values = np.array(era_for_time[j,0,:]) # get era grid values on the given date-time and channel
        regrid = griddata(input_grid, values, output_grid, method='linear') # interpolate era5 to cosmo grid using scipy griddata linear
        interpolated_data[j,0,:] = regrid
    return interpolated_data

def format_date(dt64: np.datetime64) -> str:
    """Makes date string from date time point, for saving files."""
    return to_datetime(dt64).strftime('%Y%m%d-%H%M')

def _save_datetime_file(values: np.ndarray[np.intp], variables: np.ndarray, date: np.datetime64, filepath: str):
    filename = filepath + format_date(date)
    torch.save(values, filename)

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

def interpolate_era_time_point_to_grid(i: int, era: Dataset, input_grid: np.ndarray, output_grid: np.ndarray, output_data_path: str, output_plots_path: str = None, plot_indices=[0]):
    logging.info('interpolating time point ' + format_date(era.dates[i]))
    interpolated_data = regrid(era[i,:,:,:], input_grid=input_grid, output_grid=output_grid)
    logging.info(f'writing time point { format_date(era.dates[i])} to files in path {output_data_path}')
    _save_datetime_file(interpolated_data, era.variables, era.dates[i], os.path.join(output_data_path))
    if output_plots_path and i in plot_indices:
        datestr = format_date(era.dates[i])
        logging.info(f'plotting {datestr} to {output_plots_path}')
        for j in [0]:
        #for j in range(len(era.variables)):
            var = era.variables[j]
            # plot era original
            plot_and_save_projection(input_grid[:,0], input_grid[:,1], era[i, j, 0, :], f'{output_plots_path}/{era.variables[j]}-{datestr}-era.jpg')
            # plot interpolated
            plot_and_save_projection(output_grid[:,0], output_grid[:,1], interpolated_data[j, 0, :], f'{output_plots_path}/{era.variables[j]}-{datestr}-era-interpolated.jpg')

def save_anemoi_time_point(i: int, ds: Dataset, data_output_path: str, ds_name: str, plots_output_path: str = None, plot_indices=[0]):
    _save_datetime_file(ds[i,:,:,:], ds.variables, ds.dates[i], data_output_path)
    datestr = format_date(ds.dates[i])
    if plots_output_path and i in plot_indices:
        for j,var in enumerate(ds.variables):
            plot_and_save_projection(ds.longitudes, ds.latitudes, ds[i, j, 0, :], f'{plots_output_path}/{var}-{datestr}-{ds_name}.jpg')


def _interpolate_era5_cosmo_basic(era: Dataset, cosmo: Dataset | None, intermediate_files_path: str, threaded = True, outfile_plots_path: str =None, plot_indices=[0]):
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
    if cosmo:
        assert (era.start_date == cosmo.start_date and 
                era.end_date == cosmo.end_date and 
                era.frequency == cosmo.frequency and
                era.shape[0] == cosmo.shape[0]), "ERA and COSMO date ranges or frequencies do not align."
    input_grid = np.column_stack((era.longitudes, era.latitudes)) # stack lon-lat columns of era5 points
    output_grid = None
    if cosmo:
        output_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes)) # stack lon-lat column of cosmo points
    else:
        cosmo_latlon = torch.load(os.path.join(intermediate_files_path, 'info', 'cosmo-lat-lon'), weights_only=False)
        output_grid = np.column_stack((cosmo_latlon[:,1], cosmo_latlon[:,0]))

    dates = range(era.shape[0])
    
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

### Main method 1: Interpolate ERA grid
def _interpolate_era5_to_grid(infile_era: str, output_grid: np.ndarray, output_path: str, plot_indices=[0]):
    # read data
    lats = output_grid[:,1]
    lons = output_grid[:,0]
    min_lat = min(lats) - ERA_MARGIN_DEGREES
    max_lat = max(lats) + ERA_MARGIN_DEGREES
    min_lon = min(lons) - ERA_MARGIN_DEGREES
    max_lon = max(lons) + ERA_MARGIN_DEGREES
    area=(max_lat, min_lon, min_lat, max_lon)
    logging.info(f'projecting onto era area {area}')
    era = read_anemoi_ds(infile_era, area = area)
    logging.info('Successfully read input')
    
    # Output stats and grid
    save_anemoi_stats(era, os.path.join(output_path, "info/era-stats"))
    save_anemoi_latlon_grid(era, os.path.join(output_path, "info/era-lat-lon"))

    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_era, os.path.join(output_path, "info/era.yaml"))

    input_grid = np.column_stack((era.longitudes, era.latitudes))

    for i in range(len(era.dates)):
        interpolate_era_time_point_to_grid(i, era, input_grid, output_grid,
                                           os.path.join(output_path, "era-interpolated"),
                                           os.path.join(output_path, "plots"),
                                           plot_indices)


### Part 2: Save COSMO data
def _save_cosmo_as_torch(infile_cosmo: str, outfile_data_path: str, outfile_plots_path: str = None, plot_indices=[0]):
    cosmo = read_anemoi_ds(infile_cosmo)
    save_anemoi_stats(cosmo, os.path.join(outfile_data_path, "info/cosmo-stats"))
    save_anemoi_latlon_grid(cosmo, os.path.join(outfile_data_path, "info/cosmo-lat-lon"))
    cosmo_output_path = os.path.join(outfile_data_path, "cosmo")
    for i in range(len(cosmo.dates)):
        save_anemoi_time_point(i, cosmo, data_output_path=cosmo_output_path, outfile_plots_path=outfile_plots_path, plot_incides=[0])


def interpolate_era5_to_cosmo_and_save(infile_era: str, infile_cosmo: str, outfile_data_path: str, threaded=True, outfile_plots_path: str = None, plot_indices=[0]):
    """Read both ERA and COSMO data and perform basic interpolation. Save output into Pytorch format, and (optionally) plot
    ERA, COSMO, and interpolated data.

    Parameters:
    infile_era: str
        Local file path to ERA5 data
    infile_cosmo: str 
        Local file path to COSMO2 data. Can be a lat/lon grid or a .zarr file.
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
    era = None
    cosmo = None
    if infile_cosmo.endswith('yaml'):
        era, cosmo = _read_era5_cosmo(infile_era, infile_cosmo)
        save_anemoi_stats(cosmo, os.path.join(outfile_data_path, "info/cosmo-stats"))
        save_anemoi_latlon_grid(cosmo, os.path.join(outfile_data_path, "info/cosmo-lat-lon"))
        shutil.copy(infile_cosmo, os.path.join(outfile_data_path, "info/cosmo.yaml"))
    else:
        cosmo_latlon = torch.load(infile_cosmo, weights_only=False)
        lats = cosmo_latlon[:,0]
        lons = cosmo_latlon[:,1]
        min_lat = min(lats) - ERA_MARGIN_DEGREES
        max_lat = max(lats) + ERA_MARGIN_DEGREES
        min_lon = min(lons) - ERA_MARGIN_DEGREES
        max_lon = max(lons) + ERA_MARGIN_DEGREES
        area=(max_lat, min_lon, min_lat, max_lon)
        logging.info(f'projecting onto era area {area}')
        era = read_era5_anemoi(infile_era, area = area)

    logging.info('Successfully read input')

    # Output stats and grid
    save_anemoi_stats(era, os.path.join(outfile_data_path, "info/era-stats"))
    save_anemoi_latlon_grid(era, os.path.join(outfile_data_path, "info/era-lat-lon"))

    # Copy the .yaml files over for recording purposes
    shutil.copy(infile_era, os.path.join(outfile_data_path, "info/era.yaml"))

    # generate interpolated data
    _interpolate_era5_cosmo_basic(era, cosmo, outfile_data_path, threaded=threaded, outfile_plots_path=outfile_plots_path, plot_indices=plot_indices)

    
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
