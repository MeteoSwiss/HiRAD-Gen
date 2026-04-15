These are instructions from Yoël Zérah for using sample config.

All necessary data except the datasets can be downloaded at : https://lumidata.eu/465002468:share-quickstart-era-to-cerra-downscaling-quickstart/index.html

# Datasets
## Cropped ERA5
ERA5 is a global reanalysis, whereas CERRA is regional. To perform pure downscaling at single dates, all global datapoints aren't relevant. As such, to save on storage and compute, the input low resolution ERA5 dataset is reduced to a domain around the target CERRA domain.
anemoi-datasets handles the cropping of a dataset on a spatial domain using a mask era5_to_cerra_cropping_mask.npz (provided, but see below for generation).
Using this mask to generate a cropped dataset, with the recipe :
```
yaml
name: caerraml-ea-an-oper-atos-n320crop2cerra-1991-2023-3h-v0

description: |
  This recipe subsamples the 1h full-resolution ERA5 dataset to 3h and crops it to the CERRA region + 100km buffer, for the purposes of downscaling.

dates:
  end: '2023-12-31T23:59:00'
  frequency: 3h
  start: '1983-01-01T00:00:00'

input:
    pipe: #default era5 variables and crops
        - anemoi-dataset:
                dataset: /home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-n320-1979-2023-1h-v1.zarr
                select: # all era5 variables you are interested in
                    - 10u
                    - 10v
                    - 2t
                    - sp
                    # Constants / forcings
                    - sin_julian_day
                    - sin_latitude
                    - sin_local_time
                    - sin_longitude
                    - cos_julian_day
                    - cos_latitude
                    - cos_local_time
                    - cos_longitude
                    - insolation
        - regrid:
            mask: /ec/res4/scratch/fra4433/CAERRA/datasets/suppl/era2cerra_mask_100km.npz
```            
called with the command :
```
bash
anemoi-dataset create path/to/above/recipe.yaml path/to/output.zarr
```

Depending on the number of dates, generating the dataset will require a prohibitive amount of memory. See https://anemoi.readthedocs.io/projects/datasets/en/latest/building/incremental.html#creating-a-dataset-incrementally for more details on generating a large dataset.

### Cropping mask
This mask can be generated using anemoi-transform :
```
bash
anemoi-transform make-regrid-file global-on-lam-mask --global-grid global.[npz,grib,nc] --lam-grid local.[npz,grib,nc] --output output.npz --distance <N_km>
```
With:
```
- --global-grid a file containing the lat/lon grid of the global data to be cropped
- --lam-grid a file containing the lat/lon grid of the target domain that the global grid is cropped around.
- --output is the path to the .npz cropping mask file, which contains the subset of lat/lon points of the global grid to be selected.
- --distance is a padding parameter that controls the distance (in km) around the target domain should be included in the cropped dataset.
```
### Data grid
The lam and global grids can be easily computed with anemoi-dataset:
```
python
from anemoi.datasets import 
ds_global = open_dataset("path/to/global/dataset")
np.savez("path/to/global/grid/file.npz, latitudes=ds_global.latitudes, longitudes= ds_global.longitudes)
ds_lam = open_dataset("path/to/lam/dataset")
np.savez("path/to/lam/grid/file.npz, latitudes=ds_lam.latitudes, longitudes= ds_lam.longitudes)
```

## CERRA dataset
A 3-hourly CERRA zarr source can be found at the following: /home/mlx/ai-ml/datasets/cerra-rr-an-oper-se-al-ec-mars-5p5km-1985-2023-3h-v2.zarr

# Model
## Interpolation matrices
in anemoi-core/ds-collab, interpolation matrices between low and high resolution grids are required :
- to use the 1-encoder architecture, that takes as input an interpolated low-resolution data to high-resolution.
- to predict, if desired, residual variables (i.e. predict y-x instead of y).
Such interpolation matrices can be computed :
- using a custom script (not provided) that uses input and output grids
- using the MIR library: https://www.ecmwf.int/en/newsletter/152/computing/new-ecmwf-interpolation-package-mir

Interpolation matrices matching cropped ERA5 and CERRA will be provided.
## Residual statistics
To apply normalization to residual variables, their statistics are necessary (mean, std, min, max). This can be performed using a custom script looping over high and low resolution samples, interpolating them and computing the stats of the difference. A residual statistics file is provided.ECMWFThe new ECMWF interpolation package MIR  To use ECMWF forecasts, users typically need to either transform the data from its original spherical harmonics representation (spectral space) into grid space (physical space) or map the data from the model’s Gaussian grid to a grid adapted to their needs. These operations are mostly carried out in user requests to the ECMWF real-time product generation system, to the MARS archiving and retrieval system, or within tools such as Metview and the ECMWF Web API.