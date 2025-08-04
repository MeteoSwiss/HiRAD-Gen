#!/bin/bash

#SBATCH --partition=postproc
#SBATCH --time=23:59:00

python src/input_data/interpolate_basic.py src/input_data/era-all.yaml src/input_data/cosmo-all.yaml /store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-all-channels/
