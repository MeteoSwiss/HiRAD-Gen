#!/bin/bash

#SBATCH --partition=postproc
#SBATCH --time=23:59:00

python src/hirad/input_data/interpolate_basic.py src/hirad/input_data/era-all.yaml src/hirad/input_data/cosmo-all.yaml /store_new/mch/msopr/hirad-gen/basic-torch/era5-cosmo-1h-all-channels/
