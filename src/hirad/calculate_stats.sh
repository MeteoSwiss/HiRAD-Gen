#!/bin/bash

#SBATCH --job-name="corrdiff-first-stage"

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
#SBATCH --output=/capstor/scratch/cscs/pstamenk/logs/calculate_stats.log
#SBATCH --error=/capstor/scratch/cscs/pstamenk/logs/calculate_stats.err

### ENVIRONMENT ####
#SBATCH -A a161

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=1

srun --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_env/hirad/bin/activate
    python src/hirad/input_data/calculate_transformed_stats.py
"