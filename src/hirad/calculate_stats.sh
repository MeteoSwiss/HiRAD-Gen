#!/bin/bash

#SBATCH --job-name="corrdiff-first-stage"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclusive


### OUTPUT ###
#SBATCH --output=./logs/calculate_stats.log

### ENVIRONMENT ####
#SBATCH -A c38

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=1

# STATS_CONFIG defaults to the actual training dataset config so the box-cox stats
# match the tp TARGET the model is trained on. (The old anemoi_era_real_stats_calc.yaml
# was set up for cp/2t, not the tp output — the likely origin of the wrong tp std.)
: "${STATS_CONFIG:=src/hirad/conf/dataset/anemoi_era_real.yaml}"
: "${NUM_WORKERS:=48}"   # parallel timestep-shard workers (node has 72 CPUs)
: "${STRIDE:=1}"         # 1 = full single-pass; set >1 to subsample for extra speed
echo "STATS_CONFIG=${STATS_CONFIG}  NUM_WORKERS=${NUM_WORKERS}  STRIDE=${STRIDE}"

srun --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python src/hirad/input_data/calculate_transformed_stats_anemoi.py \
        --config ${STATS_CONFIG} --output_dir ./outputs/transform_stats \
        --num-workers ${NUM_WORKERS} --stride ${STRIDE}
"