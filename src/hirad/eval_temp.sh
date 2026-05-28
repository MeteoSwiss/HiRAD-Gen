#!/bin/bash

#SBATCH --job-name="eval_temp"

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
#SBATCH --output=./logs/plots_temp.log

### ENVIRONMENT ####
#SBATCH -A a161

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .

    # Diurnal cycle of 2m temperature
    # python src/hirad/eval/diurnal_cycle_temp.py --config-name=${CONFIG_NAME}
"
