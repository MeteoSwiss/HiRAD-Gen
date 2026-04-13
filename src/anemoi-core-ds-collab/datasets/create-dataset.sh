#!/bin/bash

anemoi-datasets init --overwrite /home/che8066/anemoi-downscaling/HiRAD-Gen/src/anemoi-core-ds-collab/datasets/cerra-cropped.yaml /perm/che8066/cerra-rr-an-oper-se-al-ec-mars-5p5km-2020-2023-3h-v2.zarr
echo 'loading'
for i in $(seq 1 100);
do
    echo $i
    anemoi-datasets load /perm/che8066/cerra-rr-an-oper-se-al-ec-mars-5p5km-2020-2023-3h-v2.zarr --parts $i/100
done
echo 'finalising'
anemoi-datasets finalise /perm/che8066/cerra-rr-an-oper-se-al-ec-mars-5p5km-2020-2023-3h-v2.zarr
