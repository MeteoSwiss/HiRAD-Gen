#!/bin/bash

anemoi-datasets init era5-cosmo-cropped.yaml /capstor/scratch/cscs/mmcgloho/datasets/era-n320crop2cosmo-2015-2020-1h-v0.zarr
echo 'loading'
for i in $(seq 1 20);
do
    echo $i
    anemoi-datasets load /capstor/scratch/cscs/mmcgloho/datasets/era-n320crop2cosmo-2015-2020-1h-v0.zarr --parts $i/20
done
echo 'finalising'
anemoi-datasets finalise /capstor/scratch/cscs/mmcgloho/datasets/era-n320crop2cosmo-2015-2020-1h-v0.zarr

