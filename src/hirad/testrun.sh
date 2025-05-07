#!/bin/bash

#SBATCH --job-name="testrun"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=/scratch/mch/pstamenk/logs/regression_test.log
#SBATCH --error=/scratch/mch/pstamenk/logs/regression_test.err

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=ENV

# Get number of physical cores using Python
PHYSICAL_CORES=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")
# Use SLURM_NTASKS (number of processes to be launched by torchrun)
LOCAL_PROCS=${SLURM_NTASKS_PER_NODE:-1}
# Compute threads per process
OMP_THREADS=$(( PHYSICAL_CORES / LOCAL_PROCS ))
export OMP_NUM_THREADS=$OMP_THREADS
echo "Node: $(hostname)"
echo "Physical cores: $PHYSICAL_CORES"
echo "Local processes: $LOCAL_PROCS"
echo "Setting OMP_NUM_THREADS=$OMP_NUM_THREADS"

# activate conda env
CONDA_ENV=train
source /users/pstamenk/.bashrc
mamba activate $CONDA_ENV

# python src/hirad/training/train.py --config-name=training_era_cosmo_testrun.yaml
torchrun --nproc-per-node=$LOCAL_PROCS src/hirad/training/train.py --config-name=training_era_cosmo_regression.yaml