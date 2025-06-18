#!/bin/bash
conda init
conda activate anemoi2
#python src/input_data/interpolate_basic.py src/input_data/era-1h.yaml src/input_data/cosmo-1h.yaml /store_new/mch/msopr/hirad-gen/basic-torch/trim_19_1h/
python src/input_data/interpolate_basic.py src/input_data/era-test.yaml src/input_data/cosmo-test.yaml /store_new/mch/msopr/hirad-gen/basic-torch/trim_19_thread/