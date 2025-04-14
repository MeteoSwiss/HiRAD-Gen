import sys

from anemoi.datasets import open_dataset
from anemoi.datasets.data.dataset import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
from scipy.interpolate import griddata

# Default paths for input data sets
ERA_PATH = '/scratch/mch/apennino/data/aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr' # path in Balfrin
COSMO_PATH = '/scratch/mch/fzanetta/data/anemoi/datasets/mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr' # path in Balfrin

def _read_input(era_filepath: str, cosmo_path: str, bound_to_cosmo_area=True) -> tuple[Dataset, Dataset]:
    """
    Read both ERA and COSMO data, optionally bounding to the COSMO data area, and return the 2m
    temperature values for the time range under COSMO.
    """
    # trim edge removes boundary
    cosmo = open_dataset(COSMO_PATH, select="2t", trim_edge=20)
    # Get date and area bounds of COSMO
    start_date = cosmo.metadata()['start_date']
    end_date = cosmo.metadata()['end_date']
    # load era5 2m-temperature in the time-range of cosmo
    # area = N, W, S, E
    if bound_to_cosmo_area:
        min_lat_cosmo = min(cosmo.latitudes)
        max_lat_cosmo = max(cosmo.latitudes)
        min_lon_cosmo = min(cosmo.longitudes)
        max_lon_cosmo = max(cosmo.longitudes)
        era = open_dataset(ERA_PATH, select="2t", start=start_date, end=end_date,
                        area=(max_lat_cosmo, min_lon_cosmo, min_lat_cosmo, max_lon_cosmo))
    else:
        era = open_dataset(ERA_PATH, select="2t", start=start_date, end=end_date)
    return (era, cosmo)

def _interpolate_basic(era: Dataset, cosmo: Dataset) -> np.ndarray[np.intp]:
    """Perform simple interpolation from ERA5 to COSMO grid for the first data point in the series."""
    grid = np.column_stack((era.longitudes, era.latitudes)) # stack lon-lat columns of era5 points
    values = np.array(era[0,0,0,:]) # get era grid 2m-temperature values on the first avaialble date-time

    interp_grid = np.column_stack((cosmo.longitudes, cosmo.latitudes)) # stack lon-lat column of cosmo points

    return griddata(grid,values,interp_grid,method='linear') # interpolate era5 to cosmo grid using scipy griddata linear

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

def _plot_projection(longitudes: np.array, latitudes: np.array, values: np.array, filename: str):
    """Plot observed or interpolated data in a scatter plot."""
    fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    p = ax.scatter(x=longitudes, y=latitudes, c=values)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.colorbar(p, label="K", orientation="horizontal")
    plt.savefig(filename)

def interpolate_and_save(infile_era: str, infile_cosmo: str, outfile_torch: str, outfile_plots_prefix: str = None):
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
    era, cosmo = _read_input(ERA_PATH, COSMO_PATH, bound_to_cosmo_area=True)

    interpolated = _interpolate_basic(era, cosmo)

    if outfile_plots_prefix:
        # plot era original
        _plot_projection(era.longitudes, era.latitudes, era[0, 0, 0, :], outfile_plots_prefix + "-era.jpg")

        # plot cosmo original
        _plot_projection(cosmo.longitudes, cosmo.latitudes, cosmo[0, 0, 0, :], outfile_plots_prefix + "-cosmo.jpg")

        #plot interpolated era5
        _plot_projection(cosmo.longitudes, cosmo.latitudes, interpolated, outfile_plots_prefix + "-interpolated.jpg")

    _save_interpolation(interpolated, outfile_torch)


def main():
    interpolate_and_save(ERA_PATH, COSMO_PATH, "interpolated.torch", "temperature-2m")

if __name__ == "__main__":
    main()


