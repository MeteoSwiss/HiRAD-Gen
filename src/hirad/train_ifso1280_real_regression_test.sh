#!/bin/bash

#SBATCH --job-name="ifso1280-real-regression-test"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/training_ifso1280_real_regression_test.log

### ENVIRONMENT ####
#SBATCH -A c38

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=72

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies
    python src/hirad/training/train.py --config-name=training_ifso1280_real_regression_test.yaml
"
