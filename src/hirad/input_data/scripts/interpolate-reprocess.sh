#!/bin/bash

#SBATCH --time=12:00:00

echo 'activating env'
source /users/mmcgloho/interpolate-env-ssp/bin/activate
echo 'running'
#python src/hirad/input_data/process_torch_to_numpy.py /store_new/mch/msopr/hirad-gen/basic-torch/era5-realch1/v1.0/era-copernicus-interpolated/ /iopsstor/scratch/cscs/mmcgloho/basic-numpy/era5-realch1/v1.0-channel-subset/era-copernicus-interpolated/
#python src/hirad/input_data/process_torch_to_numpy.py /capstor/scratch/cscs/mmcgloho/basic-torch/era5-realch1/v1.0/era-copernicus-interpolated/ /iopsstor/scratch/cscs/mmcgloho/basic-numpy/era5-realch1/v1.0-channel-subset/era-copernicus-interpolated/
python src/hirad/input_data/check-fileload.py /iopsstor/scratch/cscs/mmcgloho/basic-numpy/era5-realch1/v1.0-channel-subset/realch1/