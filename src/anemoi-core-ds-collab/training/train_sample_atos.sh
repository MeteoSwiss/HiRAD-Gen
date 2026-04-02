#!/bin/bash
#SBATCH --qos=ng
#SBATCH --gpus=1

source /hpcperm/che8066/envs/anemoi-downscaling/bin/activate

ANEMOI_BASE_SEED=1
SLURM_GPUS_PER_NODE=1
SLURM_NNODES=1

anemoi-training train --config-name=sample_training_config_atos
