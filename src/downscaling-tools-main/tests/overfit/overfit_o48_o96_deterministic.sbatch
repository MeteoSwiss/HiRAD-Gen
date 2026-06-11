#!/bin/bash --login
#SBATCH --job-name=overfit_o48_1s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/overfit/%x_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/overfit/%x_%j.err

set -euxo pipefail

# ── Environment ──
source /home/ecm5702/dev/.ds-dyn/bin/activate
export ANEMOI_BASE_SEED=42
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=""
export DEV=/home/ecm5702/dev/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export ANEMOI_CONFIG_DIR=/home/ecm5702/dev/anemoi-config
export ANEMOI_CONFIG_NAME=o48_o96
export ANEMOI_CONFIG_PATH=/home/ecm5702/dev/anemoi-config
# Use pre-merge training code (compat with user config schema)
export PYTHONPATH="/ec/res4/scratch/ecm5702/aifs/anemoi-core-compat-training/models/src:/ec/res4/scratch/ecm5702/aifs/anemoi-core-compat-training/training/src:${PYTHONPATH:-}"
# Pin all dataloader iterations to the same date index (true 1-sample overfit)
export ANEMOI_OVERFIT_ONE_SAMPLE=1

LANE=o48_o96
RUN_NAME="overfit_1s_${LANE}_${SLURM_JOB_ID}"
WORK_DIR="/ec/res4/scratch/ecm5702/aifs/overfit_tests/${RUN_NAME}"
PRED_DIR="${WORK_DIR}/predictions"
SPECTRA_DIR="${WORK_DIR}/spectra"
TOOLS=/home/ecm5702/dev/downscaling-tools
VALIDATE="${TOOLS}/tests/overfit/validate_overfit.py"

mkdir -p "${WORK_DIR}" "${PRED_DIR}" "${SPECTRA_DIR}"
mkdir -p /home/ecm5702/dev/jobscripts/outputs/overfit

hostname -s
echo "LANE=${LANE} RUN_NAME=${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home/ecm5702/dev

# ── Step 1: Train (probabilistic, 2t only, 1 sample pinned) ──
echo ""
echo "========================================"
echo "STEP 1: Training probabilistic overfit (2t only, 1 pinned sample × 5000 epochs)"
echo "========================================"

srun anemoi-training train \
  --config-name=o48_o96 \
  dataloader=overfit_o48_o96_2t \
  diagnostics.log.mlflow.run_name="${RUN_NAME}" \
  diagnostics.log.mlflow.enabled=False \
  diagnostics.log.tensorboard.enabled=False \
  diagnostics.enable_checkpointing=True \
  diagnostics.checkpoint.every_n_epochs.save_frequency=5000 \
  diagnostics.checkpoint.every_n_train_steps.save_frequency=5000 \
  diagnostics.checkpoint.every_n_minutes.save_frequency=999 \
  hardware.num_gpus_per_model=1 \
  hardware.num_gpus_per_node=1 \
  dataloader.limit_batches.training=1 \
  dataloader.limit_batches.validation=0 \
  dataloader.num_workers.training=2 \
  training.max_epochs=5000 \
  training.max_steps=5000 \
  training.num_sanity_val_steps=0 \
  training.lr.rate=1e-3 \
  training.lr.warmup=100 \
  training.lr.iterations=5000 \
  model.model.diffusion.log_normal_mean=-1.2 \
  model.model.diffusion.log_normal_std=1.2

echo "Training done."

# ── Step 2: Find and convert checkpoint ──
echo ""
echo "========================================"
echo "STEP 2: Converting checkpoint to inference format"
echo "========================================"

echo "Looking for checkpoint in: ${OUTPUT}/checkpoint/"
RUN_UUID=$(ls -tc "${OUTPUT}/checkpoint/" | head -1)
CKPT_DIR="${OUTPUT}/checkpoint/${RUN_UUID}"
echo "Found checkpoint dir: ${CKPT_DIR}"

TRAIN_CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -o -name "anemoi-*.ckpt" | grep -v inference | sort | tail -1)
echo "Training checkpoint: ${TRAIN_CKPT}"

INF_CKPT="${CKPT_DIR}/inference-$(basename ${TRAIN_CKPT})"
python -m anemoi.training checkpoint inference \
  --input "${TRAIN_CKPT}" \
  --output "${INF_CKPT}"
echo "Inference checkpoint: ${INF_CKPT}"

# Keep old models AND training on PYTHONPATH for inference
export PYTHONPATH="/ec/res4/scratch/ecm5702/aifs/anemoi-core-compat-training/models/src:/ec/res4/scratch/ecm5702/aifs/anemoi-core-compat-training/training/src"

# ── Step 3: Run inference (probabilistic model, 40-step Heun, sigma_max=80) ──
echo ""
echo "========================================"
echo "STEP 3: Running inference (40-step Heun, sigma_max=80, S_churn=0)"
echo "========================================"

cd "${TOOLS}"
python -m manual_inference.prediction.predict from-dataloader \
  --debug-from-dataloader \
  --name-ckpt "${TRAIN_CKPT}" \
  --ckpt-root "" \
  --device cuda \
  --idx 0 \
  --n-samples 1 \
  --members 0 \
  --validation-frequency "50h" \
  --precision fp32 \
  --output-weather-states "2t" \
  --slim-output \
  --extra-args-json '{"schedule_type":"karras","num_steps":40,"sigma_max":80.0,"sigma_min":0.002,"rho":7.0,"sampler":"heun","S_churn":0,"S_min":0.75,"S_max":80.0,"S_noise":1.0}' \
  --allow-existing-output-dir \
  --out "${PRED_DIR}/predictions_overfit.nc"

echo "Predictions written to: ${PRED_DIR}"
ls -lh "${PRED_DIR}"/*.nc

# Quick RMSE diagnostic
python3 -c "
import xarray as xr, numpy as np
ds = xr.open_dataset('${PRED_DIR}/predictions_overfit.nc')
yp = ds['y_pred'].sel(weather_state='2t').values.squeeze()
y  = ds['y'].sel(weather_state='2t').values.squeeze()
xi = ds['x_interp'].sel(weather_state='2t').values.squeeze()
print(f'RMSE(model, truth):  {np.sqrt(np.mean((yp-y)**2)):.4f} K')
print(f'RMSE(interp, truth): {np.sqrt(np.mean((xi-y)**2)):.4f} K')
print(f'RMSE(model, interp): {np.sqrt(np.mean((yp-xi)**2)):.4f} K')
print(f'Corr(res_pred,res_truth): {np.corrcoef((yp-xi).flatten(),(y-xi).flatten())[0,1]:.4f}')
"

# ── Step 4: Run spectra (healpy) ──
echo ""
echo "========================================"
echo "STEP 4: Computing spectra (healpy, nside=32, lmax=95)"
echo "========================================"

export PYTHONPATH="${TOOLS}:${PYTHONPATH:-}"
python "${TOOLS}/eval/jobs/templates/predictions_dir_spectra.py" \
  --predictions-dir "${PRED_DIR}" \
  --out-dir "${SPECTRA_DIR}" \
  --run-label "${RUN_NAME}" \
  --weather-states "2t" \
  --nside 32 \
  --lmax 95 \
  --score-wavenumber-min-exclusive 0 \
  --spectra-method healpy

echo "Spectra written to: ${SPECTRA_DIR}"

# ── Step 5: Validate ──
echo ""
echo "========================================"
echo "STEP 5: Validating results"
echo "========================================"

PRED_NC="${PRED_DIR}/predictions_overfit.nc"
python "${VALIDATE}" \
  --predictions-nc "${PRED_NC}" \
  --spectra-json "${SPECTRA_DIR}/spectra_summary.json" \
  --variable 2t \
  --full-field-threshold 0.005 \
  --residual-threshold 0.05 \
  --rmse-threshold 0.1 \
  --temp-min 180 \
  --temp-max 340

echo ""
echo "Test artifacts: ${WORK_DIR}"
