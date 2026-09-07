#!/bin/bash

### OUTPUT ###
#SBATCH --output=./logs/calculate_ifso1280_real_boxcox_stats.log

export OMP_NUM_THREADS=72

pip install -e . --no-dependencies
python src/hirad/input_data/calculate_ifso1280_real_boxcox_stats.py "$@"
