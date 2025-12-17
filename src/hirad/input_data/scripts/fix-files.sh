#!/bin/bash

#clariden
regex_pattern="^era-interpolated(.*)"
for f in $(ls /capstor/scratch/cscs/mmcgloho/basic-torch/era5-cosmo-1h-all-channels/era-interpolated/);
do
    if [[ "$f" =~ $regex_pattern ]]; then
        newf=${BASH_REMATCH[1]}
        echo "newf: $newf"
        mv /capstor/scratch/cscs/mmcgloho/basic-torch/era5-cosmo-1h-all-channels/era-interpolated/$f /capstor/scratch/cscs/mmcgloho/basic-torch/era5-cosmo-1h-all-channels/era-interpolated/$newf
    fi
done