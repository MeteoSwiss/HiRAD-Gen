#!/bin/bash

uenv start pytorch
source .venv/bin/activate

python src/anemoi-core-ds-collab/datasets/compute_residuals.py \
--lres_dataset=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr \
--hres_dataset=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/mch-co2-an-archive-0p02-2015-2020-1h-v3-pl13.zarr \
--interpolation_matrix=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/era2cosmo_cropped_to_cosmo_linear.mat.npz \
--output_path=/capstor/store/cscs/pasc/c38/anemoi-downscaling-data/era_cosmo_residuals_2015_2015_2 \
--start_date=2015-11-28 --end_date=2016-01-01
