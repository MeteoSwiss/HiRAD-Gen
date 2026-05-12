#!/bin/bash --login
#SBATCH --job-name=overfit_suite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/overfit/%x_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/overfit/%x_%j.err

set -euxo pipefail

# ── Required: MODE env var ──
# Modes: backward_compat, residual_2t_det, residual_2t_prob,
#         residual_2t_tp_det, residual_2t_tp_prob
# Submit: MODE=residual_2t_det sbatch overfit_o48_o96_suite.sbatch
#    or:  sbatch --export=ALL,MODE=residual_2t_det overfit_o48_o96_suite.sbatch
MODE=${MODE:?"MODE is required (backward_compat|residual_2t_det|residual_2t_prob|residual_2t_tp_det|residual_2t_tp_prob)"}
STEPS=${STEPS:-5000}

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

# Dev anemoi-core for all modes (the code under validation)
export PYTHONPATH="/home/ecm5702/dev/anemoi-core/models/src:/home/ecm5702/dev/anemoi-core/training/src:${PYTHONPATH:-}"

LANE=o48_o96
RUN_NAME="overfit_suite_${MODE}_${SLURM_JOB_ID}"
WORK_DIR="/ec/res4/scratch/ecm5702/aifs/overfit_tests/${RUN_NAME}"
PRED_DIR="${WORK_DIR}/predictions"
SPECTRA_DIR="${WORK_DIR}/spectra"
TOOLS=/home/ecm5702/dev/downscaling-tools
VALIDATE="${TOOLS}/tests/overfit/validate_overfit.py"

mkdir -p "${WORK_DIR}" "${PRED_DIR}" "${SPECTRA_DIR}"
mkdir -p /home/ecm5702/dev/jobscripts/outputs/overfit

hostname -s
echo "MODE=${MODE} STEPS=${STEPS} LANE=${LANE} RUN_NAME=${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home/ecm5702/dev

# ── Mode-specific configuration ──
SKIP_TRAINING=0
SELECT_MATCHED_OVERRIDE=""
TRAINING_APPROACH_OVERRIDE=""
DIRECT_PREDICTION_OVERRIDE=""
OUTPUT_WEATHER_STATES=""
VALIDATE_VARIABLES=""
VALIDATE_EXTRA_ARGS=""

case "$MODE" in
  backward_compat)
    CKPT_PATH=${CKPT_PATH:?"backward_compat requires CKPT_PATH env var"}
    SKIP_TRAINING=1
    OUTPUT_WEATHER_STATES="${COMPAT_WEATHER_STATES:-2t}"
    VALIDATE_VARIABLES="${COMPAT_WEATHER_STATES:-2t}"
    ;;
  residual_2t_det)
    SELECT_MATCHED_OVERRIDE="dataloader.select_matched=[2t]"
    TRAINING_APPROACH_OVERRIDE="training.training_approach=deterministic"
    OUTPUT_WEATHER_STATES="2t"
    VALIDATE_VARIABLES="2t"
    ;;
  residual_2t_prob)
    SELECT_MATCHED_OVERRIDE="dataloader.select_matched=[2t]"
    TRAINING_APPROACH_OVERRIDE="training.training_approach=probabilistic_low_noise"
    OUTPUT_WEATHER_STATES="2t"
    VALIDATE_VARIABLES="2t"
    ;;
  residual_2t_tp_det)
    SELECT_MATCHED_OVERRIDE="dataloader.select_matched=[2t,tp]"
    TRAINING_APPROACH_OVERRIDE="training.training_approach=deterministic"
    DIRECT_PREDICTION_OVERRIDE="data.direct_prediction=[tp]"
    OUTPUT_WEATHER_STATES="2t,tp"
    VALIDATE_VARIABLES="2t,tp"
    ;;
  residual_2t_tp_prob)
    SELECT_MATCHED_OVERRIDE="dataloader.select_matched=[2t,tp]"
    TRAINING_APPROACH_OVERRIDE="training.training_approach=probabilistic_low_noise"
    DIRECT_PREDICTION_OVERRIDE="data.direct_prediction=[tp]"
    OUTPUT_WEATHER_STATES="2t,tp"
    VALIDATE_VARIABLES="2t,tp"
    ;;
  *)
    echo "ERROR: Unknown MODE='${MODE}'"
    echo "Valid: backward_compat, residual_2t_det, residual_2t_prob, residual_2t_tp_det, residual_2t_tp_prob"
    exit 1
    ;;
esac

echo "Config: SELECT_MATCHED=${SELECT_MATCHED_OVERRIDE} TRAINING_APPROACH=${TRAINING_APPROACH_OVERRIDE}"
echo "Config: DIRECT_PREDICTION=${DIRECT_PREDICTION_OVERRIDE}"
echo "Config: OUTPUT_WEATHER_STATES=${OUTPUT_WEATHER_STATES} VALIDATE_VARIABLES=${VALIDATE_VARIABLES}"

# ── Step 1: Train (overfit on 1 sample) ──
if [[ "$SKIP_TRAINING" -eq 0 ]]; then
  echo ""
  echo "========================================"
  echo "STEP 1: Training overfit (1 batch × ${STEPS} steps) — MODE=${MODE}"
  echo "========================================"

  HYDRA_OVERRIDES=(
    --config-name=o48_o96
    dataloader=overfit_o48_o96
    diagnostics.log.mlflow.run_name="${RUN_NAME}"
    diagnostics.log.mlflow.enabled=False
    diagnostics.log.tensorboard.enabled=False
    diagnostics.enable_checkpointing=True
    diagnostics.checkpoint.every_n_epochs.save_frequency=9999
    diagnostics.checkpoint.every_n_train_steps.save_frequency=${STEPS}
    diagnostics.checkpoint.every_n_minutes.save_frequency=999
    hardware.num_gpus_per_model=1
    hardware.num_gpus_per_node=1
    +dataloader.overfit_on_index=0
    dataloader.limit_batches.training=1
    dataloader.limit_batches.validation=0
    dataloader.num_workers.training=2
    training.max_epochs=${STEPS}
    training.max_steps=${STEPS}
    training.num_sanity_val_steps=0
    training.lr.rate=1e-3
    training.lr.warmup=10
    training.lr.iterations=${STEPS}
    model.model.diffusion.log_normal_mean=-1.2
    model.model.diffusion.log_normal_std=1.2
  )

  # Add mode-specific overrides
  [[ -n "$SELECT_MATCHED_OVERRIDE" ]] && HYDRA_OVERRIDES+=("$SELECT_MATCHED_OVERRIDE")
  [[ -n "$TRAINING_APPROACH_OVERRIDE" ]] && HYDRA_OVERRIDES+=("$TRAINING_APPROACH_OVERRIDE")
  [[ -n "$DIRECT_PREDICTION_OVERRIDE" ]] && HYDRA_OVERRIDES+=("$DIRECT_PREDICTION_OVERRIDE")

  srun anemoi-training train "${HYDRA_OVERRIDES[@]}"
  echo "Training done."

  # ── Step 2: Find and convert checkpoint ──
  echo ""
  echo "========================================"
  echo "STEP 2: Converting checkpoint to inference format"
  echo "========================================"

  # Extract the checkpoint dir that THIS training job created (avoid race with concurrent jobs)
  JOB_OUT="/home/ecm5702/dev/jobscripts/outputs/overfit/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
  CKPT_DIR=$(grep -oP 'Checkpoints path: \K.*' "${JOB_OUT}" | tail -1)
  if [[ -z "$CKPT_DIR" ]]; then
    echo "WARN: Could not extract checkpoint path from training log; falling back to ls -tc"
    RUN_UUID=$(ls -tc "${OUTPUT}/checkpoint/" | head -1)
    CKPT_DIR="${OUTPUT}/checkpoint/${RUN_UUID}"
  fi
  echo "Found checkpoint dir: ${CKPT_DIR}"

  TRAIN_CKPT=$(find "${CKPT_DIR}" -name "last.ckpt" -o -name "anemoi-*.ckpt" | grep -v inference | sort | tail -1)
  echo "Training checkpoint: ${TRAIN_CKPT}"

  INF_CKPT="${CKPT_DIR}/inference-$(basename ${TRAIN_CKPT})"
  if [[ -f "${INF_CKPT}" ]]; then
    echo "Inference checkpoint already exists (created by training callback): ${INF_CKPT}"
  else
    python -m anemoi.training checkpoint inference \
      --input "${TRAIN_CKPT}" \
      --output "${INF_CKPT}"
  fi
  echo "Inference checkpoint: ${INF_CKPT}"

  CKPT_PATH="${TRAIN_CKPT}"
fi

# ── Step 3: Run sampling (40-step Heun, S_churn=0) ──
echo ""
echo "========================================"
echo "STEP 3: Running diffusion sampling (40-step Heun, S_churn=0) — MODE=${MODE}"
echo "========================================"

cd "${TOOLS}"
python -m manual_inference.prediction.predict from-dataloader \
  --debug-from-dataloader \
  --name-ckpt "${CKPT_PATH}" \
  --ckpt-root "" \
  --device cuda \
  --idx 0 \
  --n-samples 1 \
  --members 0 \
  --validation-frequency "50h" \
  --precision fp32 \
  --split train \
  --output-weather-states "${OUTPUT_WEATHER_STATES}" \
  --slim-output \
  --extra-args-json '{"schedule_type":"karras","num_steps":40,"sigma_max":1000.0,"sigma_min":0.03,"rho":7.0,"sampler":"heun","S_churn":0,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}' \
  --allow-existing-output-dir \
  --out "${PRED_DIR}/predictions_overfit.nc"

echo "Predictions written to: ${PRED_DIR}"
ls -lh "${PRED_DIR}"/*.nc

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
  --weather-states "${OUTPUT_WEATHER_STATES}" \
  --nside 32 \
  --lmax 95 \
  --score-wavenumber-min-exclusive 0 \
  --spectra-method healpy

echo "Spectra written to: ${SPECTRA_DIR}"

# ── Step 5: Validate ──
echo ""
echo "========================================"
echo "STEP 5: Validating results — MODE=${MODE}"
echo "========================================"

PRED_NC="${PRED_DIR}/predictions_overfit.nc"
python "${VALIDATE}" \
  --predictions-nc "${PRED_NC}" \
  --spectra-json "${SPECTRA_DIR}/spectra_summary.json" \
  --variables "${VALIDATE_VARIABLES}"

echo ""
echo "========================================"
echo "MODE=${MODE} — Test artifacts: ${WORK_DIR}"
echo "========================================"
