#!/bin/bash

#SBATCH --job-name="eval_wind"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/plots_wind.log

### ENVIRONMENT ####
#SBATCH -A a161

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies

    # Diurnal cycle
    # python src/hirad/eval/diurnal_cycle_temp_wind.py --config-name=${CONFIG_NAME}

    # Probability of exceedance
    # python src/hirad/eval/probability_of_exceedance_wind.py --config-name=${CONFIG_NAME}
    
    # Maps
    # python src/hirad/eval/map_wind_stats.py --config-name=${CONFIG_NAME}
"