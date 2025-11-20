#!/bin/bash

#SBATCH --time=12:00:00


#srun -A a161 -t 12:00:00 --environment=modulus_env bash -c "
#    pip install -e . --no-dependencies
#    pip install anemoi.datasets
#    python src/hirad/input_data/interpolate_basic.py src/hirad/input_data/era-all.yaml src/hirad/input_data/cosmo-all.yaml /capstor/scratch/cscs/mmcgloho/datasets/processed/era5-cosmo-1h-all-channels/
#" 
pip install -e . --no-dependencies
pip install anemoi.datasets
#pip install meteodata-lab
#python src/hirad/input_data/interpolate_realch1.py \
#    src/hirad/input_data/era.yaml \
#    /capstor/store/mch/msopr/hirad-gen/basic-torch/era5-realch1/v0.2/info/realch1-lat-lon \
#    /capstor/store/mch/msopr/hirad-gen/copernicus-datasets/tp-2023-2024.nc \
#    /capstor/scratch/cscs/mmcgloho/basic-torch/era5-realch1/v1.0/
python src/hirad/input_data/interpolate_basic.py \
    $1 \
    /capstor/scratch/cscs/mmcgloho/basic-torch/era5-cosmo-1h-all-channels/info/cosmo-lat-lon \
    /capstor/scratch/cscs/mmcgloho/basic-torch/era5-cosmo-1h-all-channels/
