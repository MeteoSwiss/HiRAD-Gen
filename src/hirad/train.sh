#!/bin/bash
# Parameterized multi-node training launcher. Pick the model family via TRAIN_CONFIG
# and pass any Hydra overrides via OVERRIDES:
#
#   sbatch --export=ALL,TRAIN_CONFIG=train_dit                 src/hirad/train.sh
#   sbatch --export=ALL,TRAIN_CONFIG=train_ardit               src/hirad/train.sh
#   sbatch --export=ALL,TRAIN_CONFIG=train_corrdiff_regression src/hirad/train.sh
#   sbatch --export=ALL,TRAIN_CONFIG=train_corrdiff_diffusion  src/hirad/train.sh
#
#   # with overrides, e.g. an AR-DiT prev-frame dropout sweep:
#   sbatch --export=ALL,TRAIN_CONFIG=train_ardit,\
#OVERRIDES="training.hp.prev_hr_dropout=0.1 hydra.job.name=ardit_drop0.1" src/hirad/train.sh
#
# WARNING: sbatch splits --export on COMMAS, so any override containing commas
# (lists like channel_weights=[1,1,1,3] or times_range=[...]) gets TRUNCATED.
# For those, export the variables first and use --export=ALL:
#   export TRAIN_CONFIG=train_dit
#   export OVERRIDES="training.hp.channel_weights=[1,1,1,3] hydra.job.name=..."
#   sbatch --export=ALL src/hirad/train.sh

#SBATCH --job-name="hirad-train"

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
#SBATCH --output=./logs/train_%j.log
#SBATCH --error=./logs/train_%j.err

### ENVIRONMENT ####
#SBATCH -A c38

# Choose method to initialize dist in pytorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=1

: "${TRAIN_CONFIG:=train_dit}"
: "${OVERRIDES:=}"

echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "OVERRIDES=${OVERRIDES}"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python src/hirad/training/train.py --config-name=${TRAIN_CONFIG} ${OVERRIDES}
"
