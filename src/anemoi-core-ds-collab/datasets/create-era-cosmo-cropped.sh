#!/bin/bash

anemoi-datasets init --overwrite src/anemoi-core-ds-collab/datasets/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.yaml /store_new/mch/msopr/hirad-gen/anemoi-datasets/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr
echo 'loading'
for i in $(seq 1 100);
do
    echo $i
    anemoi-datasets load /store_new/mch/msopr/hirad-gen/anemoi-datasets/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr --parts $i/100
done
echo 'finalising'
anemoi-datasets finalise /store_new/mch/msopr/hirad-gen/anemoi-datasets/aifs-ea-an-oper-0001-mars-n320crop2cosmo-1979-2024-1h-v2-with-era51.zarr
