#!/bin/bash

pip install -e .
pip install anemoi.datasets

python src/hirad/input_data/regrid_copernicus_tp.py \
 /capstor/store/cscs/swissai/a161/datasets/copernicus/tp-2019-2020.nc \
 /capstor/store/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-linear-interpolation-full/ \
 /capstor/store/cscs/swissai/a161/era5-cosmo-1h-linear-interpolation/train/era-interpolated-with-copernicus-tp/