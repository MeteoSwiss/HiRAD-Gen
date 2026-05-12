#!/bin/bash
#SBATCH --account=c38
#SBATCH -t 3:00:00
#SBATCH --nodes=1

uenv start pytorch
source .venv/bin/activate

ANEMOI_BASE_SEED=1
SLURM_GPUS_PER_NODE=4
SLURM_NNODES=1

anemoi-training train --config-name=sample_training_config_santis
