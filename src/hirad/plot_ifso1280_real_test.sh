#!/bin/bash

#SBATCH --job-name="ifso1280-real-plot-maps"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=./logs/plot_ifso1280_real_test.log

### ENVIRONMENT ####
#SBATCH -A c38

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies
    python src/hirad/eval/plot_maps.py --config-name=plotting_ifso1280_real.yaml
"
