#!/bin/bash

#SBATCH --job-name="snapshots"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=/capstor/scratch/cscs/pstamenk/logs/snapshot.log

### ENVIRONMENT ####
#SBATCH -A a161

srun --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_env/hirad/bin/activate
    python src/hirad/eval/snapshots.py --config-name=src/hirad/conf/eval_real.yaml
"