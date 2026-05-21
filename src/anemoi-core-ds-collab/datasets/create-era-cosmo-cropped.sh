#!/bin/bash

#anemoi-datasets init --overwrite src/anemoi-core-ds-collab/datasets/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.yaml /capstor/scratch/cscs/mmcgloho/anemoi-downscaling/downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr
echo 'loading'
for i in $(seq 2 100);
do
    echo $i
    anemoi-datasets load /capstor/scratch/cscs/mmcgloho/anemoi-downscaling/downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr --parts $i/100
done
echo 'finalising'
anemoi-datasets finalise /capstor/scratch/cscs/mmcgloho/anemoi-downscaling/downscaling-data/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr
