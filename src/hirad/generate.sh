#!/bin/bash

#SBATCH --job-name="testrun"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=/capstor/scratch/cscs/pstamenk/logs/regression_generation.log
#SBATCH --error=/capstor/scratch/cscs/pstamenk/logs/regression_generation.err

### ENVIRONMENT ####
#SBATCH -A c38

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "Master node : $MASTER_ADDR"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
echo "Master address : $MASTER_ADDR"
export MASTER_ADDR
export MASTER_PORT=29500
echo "Master port: $MASTER_PORT"

# Get number of physical cores using Python
# PHYSICAL_CORES=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")
# # Use SLURM_NTASKS (number of processes to be launched by torchrun)
# LOCAL_PROCS=${SLURM_NTASKS_PER_NODE:-1}
# # Compute threads per process
# OMP_THREADS=$(( PHYSICAL_CORES / LOCAL_PROCS ))
# export OMP_NUM_THREADS=$OMP_THREADS
export OMP_NUM_THREADS=72
# echo "Physical cores: $PHYSICAL_CORES"
# echo "Local processes: $LOCAL_PROCS"
# echo "Setting OMP_NUM_THREADS=$OMP_NUM_THREADS"

# python src/hirad/training/train.py --config-name=training_era_cosmo_testrun.yaml
srun --container-writable --environment=modulus_env bash -c "
    cd HiRAD-Gen
    pip install -e . --no-dependencies
    pip install Cartopy==0.22.0
    pip install xskillscore
    python src/hirad/inference/generate.py --config-name=generate_era_cosmo.yaml
"