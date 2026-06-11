#!/bin/bash
# ============================================================================
# Master runner for the anemoi-core downscaling validation suite
#
# Submits all 5 experiments and chains validation jobs after each training.
# All experiments are independent except validation depends on its training.
#
# Usage:
#   bash run_validation_suite.sh
# ============================================================================

set -euo pipefail

TEMPLATE_DIR="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/validation"
SUBMIT_DIR="/home/ecm5702/dev/jobscripts/submit/validation_suite"
VALIDATE_SCRIPT="${TEMPLATE_DIR}/validate_checkpoint.py"
CHECKPOINT_BASE="/ec/res4/scratch/ecm5702/aifs/checkpoint"

mkdir -p "${SUBMIT_DIR}"

echo "============================================"
echo "  Anemoi-Core Downscaling Validation Suite  "
echo "============================================"
echo ""
echo "Template dir: ${TEMPLATE_DIR}"
echo "Submit dir:   ${SUBMIT_DIR}"
echo ""

# --- Submit Exp 1: 2t deterministic ---
cp "${TEMPLATE_DIR}/val_exp1_2t_deterministic.sbatch" "${SUBMIT_DIR}/"
JOB1=$(sbatch --parsable "${SUBMIT_DIR}/val_exp1_2t_deterministic.sbatch")
echo "Exp 1 (2t deterministic):   Job ${JOB1}"

# --- Submit Exp 2: 2t diffusion ---
cp "${TEMPLATE_DIR}/val_exp2_2t_diffusion.sbatch" "${SUBMIT_DIR}/"
JOB2=$(sbatch --parsable "${SUBMIT_DIR}/val_exp2_2t_diffusion.sbatch")
echo "Exp 2 (2t diffusion):       Job ${JOB2}"

# --- Submit Exp 3: 2t+tp deterministic ---
cp "${TEMPLATE_DIR}/val_exp3_2t_tp_deterministic.sbatch" "${SUBMIT_DIR}/"
JOB3=$(sbatch --parsable "${SUBMIT_DIR}/val_exp3_2t_tp_deterministic.sbatch")
echo "Exp 3 (2t+tp deterministic): Job ${JOB3}"

# --- Submit Exp 4: 2t+tp diffusion ---
cp "${TEMPLATE_DIR}/val_exp4_2t_tp_diffusion.sbatch" "${SUBMIT_DIR}/"
JOB4=$(sbatch --parsable "${SUBMIT_DIR}/val_exp4_2t_tp_diffusion.sbatch")
echo "Exp 4 (2t+tp diffusion):    Job ${JOB4}"

# --- Submit Exp 5: Full diffusion ---
cp "${TEMPLATE_DIR}/val_exp5_full_diffusion.sbatch" "${SUBMIT_DIR}/"
JOB5=$(sbatch --parsable "${SUBMIT_DIR}/val_exp5_full_diffusion.sbatch")
echo "Exp 5 (full diffusion):     Job ${JOB5}"

echo ""
echo "All training jobs submitted."
echo ""
echo "To monitor:"
echo "  squeue -u ecm5702 | grep val_"
echo ""
echo "After each job finishes, find the checkpoint under:"
echo "  ${CHECKPOINT_BASE}/"
echo ""
echo "Then run validation manually per experiment:"
echo ""
echo "  # Exp 1 (deterministic, strict threshold):"
echo "  python ${VALIDATE_SCRIPT} \\"
echo "    --checkpoint <CKPT_PATH>/last.ckpt \\"
echo "    --mode deterministic --rmse-threshold 1e-4"
echo ""
echo "  # Exp 2 (diffusion, Karras σ_max=100, 20 steps):"
echo "  python ${VALIDATE_SCRIPT} \\"
echo "    --checkpoint <CKPT_PATH>/last.ckpt \\"
echo "    --mode diffusion --sigma-max 100 --num-steps 20 --rmse-threshold 0.1"
echo ""
echo "  # Exp 3 (deterministic, strict threshold):"
echo "  python ${VALIDATE_SCRIPT} \\"
echo "    --checkpoint <CKPT_PATH>/last.ckpt \\"
echo "    --mode deterministic --rmse-threshold 1e-4"
echo ""
echo "  # Exp 4 (diffusion, Karras σ_max=100, 20 steps):"
echo "  python ${VALIDATE_SCRIPT} \\"
echo "    --checkpoint <CKPT_PATH>/last.ckpt \\"
echo "    --mode diffusion --sigma-max 100 --num-steps 20 --rmse-threshold 0.1"
echo ""
echo "  # Exp 5: just check the loss in the training log — no checkpoint validation needed."
echo ""
echo "Job IDs: ${JOB1} ${JOB2} ${JOB3} ${JOB4} ${JOB5}"
