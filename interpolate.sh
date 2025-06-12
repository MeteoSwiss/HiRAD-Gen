#!/bin/bash
conda init
conda activate regridding
python src/input_data/interpolate_basic.py src/input_data/era.yaml src/input_data/cosmo.yaml /store_new/mch/msopr/hirad-gen/basic-torch/trim_19_full/
