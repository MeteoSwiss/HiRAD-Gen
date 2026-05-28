#!/bin/bash

#SBATCH --job-name="eval_precip"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/plots_precip_bias.log

### ENVIRONMENT ####
#SBATCH -A c38

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .

    # Diurnal cycle
    # python src/hirad/eval/diurnal_cycle_precip_mean_wet-hour.py --config-name=${CONFIG_NAME}
    # python src/hirad/eval/diurnal_cycle_precip_high_percentiles.py --config-name=${CONFIG_NAME}

    # Histograms
    # python src/hirad/eval/hist.py --config-name=${CONFIG_NAME}
    # python src/hirad/eval/probability_of_exceedance.py --config-name=${CONFIG_NAME}

    # QQ
    # python -m hirad.eval.bias_by_percentile_precip --config-name=${CONFIG_NAME}

    # Maps
    # python src/hirad/eval/map_precip_stats.py --config-name=${CONFIG_NAME}
    # python -m hirad.eval.diurnal_cycle_precip_maps --config-name=${CONFIG_NAME}
"