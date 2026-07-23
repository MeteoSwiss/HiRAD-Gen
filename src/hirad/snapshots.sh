#!/bin/bash

#SBATCH --job-name="snapshots"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/snapshot.log

### ENVIRONMENT ####
#SBATCH -A c38

# EVAL_CONFIG defaults to the standard eval config; override per run, e.g.:
#   sbatch --export=ALL,EVAL_CONFIG=src/hirad/conf/my_snapshot_cfg.yaml src/hirad/snapshots.sh
: "${EVAL_CONFIG:=src/hirad/conf/eval_real.yaml}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python src/hirad/eval/snapshots.py --config-name=${EVAL_CONFIG}
"