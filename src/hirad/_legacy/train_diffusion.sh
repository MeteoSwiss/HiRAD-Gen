#!/bin/bash

#SBATCH --job-name="corrdiff-second-stage"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/train_diffusion.log
#SBATCH --error=./logs/train_diffusion.err

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

export OMP_NUM_THREADS=1

# python src/hirad/training/train.py --config-name=training_era_cosmo_testrun.yaml
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/training/train.py --config-name=training_era_real_diffusion_patched.yaml
"