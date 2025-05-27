#!/bin/bash

#SBATCH --job-name="testrun"

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
#SBATCH --output=/iopsstor/scratch/cscs/pstamenk/logs/diffusion.log
#SBATCH --error=/iopsstor/scratch/cscs/pstamenk/logs/diffusion.err

### ENVIRONMENT ####
#SBATCH --uenv=pytorch/v2.6.0:/user-environment
#SBATCH --view=default
#SBATCH -A a-a122

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

# Get number of physical cores using Python
PHYSICAL_CORES=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")
LOCAL_PROCS=${SLURM_NTASKS_PER_NODE:-1}
# Compute cores per process
OMP_THREADS=$(( PHYSICAL_CORES / LOCAL_PROCS ))
export OMP_NUM_THREADS=$OMP_THREADS

# python src/hirad/training/train.py --config-name=training_era_cosmo_testrun.yaml
srun bash -c "
    . ./train_env/bin/activate
    python src/hirad/training/train.py --config-name=training_era_cosmo_diffusion.yaml
"