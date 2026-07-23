#!/bin/bash
# Parameterized generation job (mirrors eval_precip_job.sh).
# Submit via submit/sampler_sweep.sh, or directly:
#   sbatch --export=ALL,GEN_SCRIPT=src/hirad/inference/generate.py,\
#CONFIG_NAME=generate,\
#OVERRIDES="generation=dit sampler.params.sigma_max=80 hydra.job.name=dit_edm_smax80_n32" \
#     src/hirad/generate_job.sh

#SBATCH --job-name="hirad_gen"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/gen_%j.log

### ENVIRONMENT ####
#SBATCH -A c38

export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

: "${GEN_SCRIPT:=src/hirad/inference/generate.py}"
: "${OVERRIDES:=}"

echo "GEN_SCRIPT=${GEN_SCRIPT}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "OVERRIDES=${OVERRIDES}"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python ${GEN_SCRIPT} --config-name=${CONFIG_NAME} ${OVERRIDES}
"
