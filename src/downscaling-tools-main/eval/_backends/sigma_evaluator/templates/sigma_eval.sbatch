#!/bin/bash
# Sigma evaluator — single parameterized template for all 4 downscaling lanes.
#
# Usage:
#   1. Copy this file to ~/dev/jobscripts/submit/<date>/sigma_eval_<RUN_ID>.sbatch
#   2. Fill in the lane config block below
#   3. sbatch sigma_eval_<RUN_ID>.sbatch
#
# Lane config — edit these for each run:
LANE="o96_o320"                   # o48_o96 | o96_o320 | o320_o1280 | o1280_o2560
CHECKPOINT_REF=""                 # MLflow run ID (32-char hex)
CKPT_NAME=""                      # e.g. anemoi-by_epoch-epoch_021-step_100000.ckpt
RUN_ID=""                         # short label for output file naming
STACK_FLAVOR="new"                # new | old
SIGMAS="1,5,10,100"
N_SAMPLES=10
VALIDATION_FREQUENCY="50h"

# ---------- Derived from LANE ----------
case $LANE in
    o320_o1280|o1280_o2560)
        NGPU=4
        ;;
    *)
        NGPU=1
        ;;
esac

#SBATCH --job-name=sigma-${LANE}-${RUN_ID}
#SBATCH --nodes=1
#SBATCH --qos=ng
#SBATCH --gpus-per-node=${NGPU}
#SBATCH --ntasks-per-node=${NGPU}
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/%x-%j.out

set -eux

# ---------- Host validation ----------
HOSTNAME_SHORT=$(hostname -s)
case $LANE in
    o320_o1280|o1280_o2560)
        if [[ ! "$HOSTNAME_SHORT" =~ ^ag ]]; then
            echo "ERROR: Lane $LANE requires AG host (got $HOSTNAME_SHORT)" >&2
            exit 1
        fi
        ;;
esac

# ---------- Venv selection ----------
case $HOSTNAME_SHORT in
    ag*)
        case $STACK_FLAVOR in
            new)  VENV=/home/ecm5702/dev/.ds-ag/bin/activate ;;
            old)  VENV=/home/ecm5702/dev/.ds-ag/bin/activate ;;
        esac
        ;;
    *)
        case $STACK_FLAVOR in
            new)  VENV=/home/ecm5702/dev/.ds-dyn/bin/activate ;;
            old)  VENV=/home/ecm5702/dev/.ds-dyn/bin/activate ;;
        esac
        ;;
esac
source "$VENV"

# ---------- Environment ----------
export ANEMOI_BASE_SEED=756
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEV=/home/ecm5702/dev/
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

WORKDIR=/home/ecm5702/dev/downscaling-tools
cd "$WORKDIR"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

# ---------- Output path ----------
OUT_CSV="/home/ecm5702/scratch/eval/scoreboards/sigma/${RUN_ID}_sigma_eval.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# ---------- Run ----------
srun python -u -m eval._backends.sigma_evaluator.run_sigma_evaluator \
    --name_exp "$CHECKPOINT_REF" \
    --name_ckpt "$CKPT_NAME" \
    --out_csv "$OUT_CSV" \
    --sigmas "$SIGMAS" \
    --n_samples "$N_SAMPLES" \
    --validation_frequency "$VALIDATION_FREQUENCY" \
    --num-gpus-per-model "$NGPU" \
    --device auto

echo "Done. Output: $OUT_CSV"
