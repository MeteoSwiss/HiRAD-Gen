#!/bin/bash --login
# ============================================================================
# Validation Exp 4: 2t + tp (direct prediction) diffusion overfit
#
# PURPOSE: Full two-variable diffusion test. 2t (residual) + tp (direct pred)
#          with probabilistic_low_noise, overfit 1 sample for 2000 steps.
# PASS CRITERION: Karras sampling (σ_max=100, 20 steps) → RMSE < 0.1 on 2t+tp.
# ============================================================================

#SBATCH --job-name=val_exp4_2ttp_diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=ng
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/%x_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/%x_%j.err

set -euxo pipefail

# ─── Environment ────────────────────────────────────────────────────────────
HOST_SHORT=$(hostname -s)
if [[ "$HOST_SHORT" == ag* ]]; then
  source /home/ecm5702/dev/.ds-ag/bin/activate
elif [[ "$HOST_SHORT" == ac* ]]; then
  source /home/ecm5702/dev/.ds-dyn/bin/activate
else
  echo "ERROR: Unknown host $HOST_SHORT" && exit 1
fi
export ANEMOI_BASE_SEED=756
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=""
export DEV=/home/ecm5702/dev/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

# ─── Diagnostics ────────────────────────────────────────────────────────────
hostname -s
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
git -C /home/ecm5702/dev/anemoi-core branch --show-current
which python
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home/ecm5702/dev

RUN_NAME="val_exp4_2ttp_diff_${SLURM_JOB_ID}"

# ─── Train ──────────────────────────────────────────────────────────────────
srun anemoi-training train \
  --config-name=o48_o96 \
  data=downscaling_o48_2t_tp \
  dataloader=o48_o96_2t_tp \
  diagnostics.log.mlflow.run_name="${RUN_NAME}" \
  hardware.num_gpus_per_model=1 \
  hardware.num_gpus_per_node=1 \
  training.max_steps=2000 \
  training.lr.rate=0.001 \
  training.lr.iterations=2000 \
  training.lr.warmup=10 \
  training.lr.min=1e-5 \
  training.training_approach=probabilistic_low_noise \
  model.model.diffusion.log_normal_mean=0 \
  model.model.diffusion.log_normal_std=3.5 \
  '+dataloader.overfit_on_index=0' \
  dataloader.batch_size.training=1 \
  dataloader.num_workers.training=1 \
  dataloader.num_workers.validation=1 \
  dataloader.limit_batches.validation=1 \
  '+diagnostics.check_val_every_n_epoch=50' \
  '~training.variable_groups.pl.param' \
  '+training.variable_groups.pl.is_pressure_level=True'
