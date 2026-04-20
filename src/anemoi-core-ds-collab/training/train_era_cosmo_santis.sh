#!/bin/bash
#SBATCH --account=c38
#SBATCH -t 12:00:00

uenv start pytorch
source .venv/bin/activate

ANEMOI_BASE_SEED=1
SLURM_GPUS_PER_NODE=4
SLURM_NNODES=1

anemoi-training train --config-name=era_cosmo_downscaling_santis
