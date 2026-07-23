#!/bin/bash

#SBATCH --job-name="corrdiff-test-second-stage"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/training_diffusion_test.log

### ENVIRONMENT ####
#SBATCH -A a161

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

# Get number of physical cores using Python
# PHYSICAL_CORES=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")
# LOCAL_PROCS=${SLURM_NTASKS_PER_NODE:-1}
# # Compute cores per process
# OMP_THREADS=$(( PHYSICAL_CORES / LOCAL_PROCS ))
# export OMP_NUM_THREADS=$OMP_THREADS
export OMP_NUM_THREADS=72

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies
    python src/hirad/training/train.py --config-name=training_era_cosmo_diffusion_test.yaml
"