#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --run-id <id> [options]

Options:
  --run-id <id>                Required run id (e.g. p25a)
  --eval-root <path>           Eval root (default: /home/ecm5702/scratch/eval)
  --input-root <path>          Bundle input root (default: /home/ecm5702/hpcperm/data/input_data/o96_o320/idalia)
  --ckpt-id <id>               Checkpoint id
  --predict-qos <qos>          Slurm QoS for predict job (default: dg; auto-switch to ng if time > 00:30:00)
  --predict-time <hh:mm:ss>    Predict walltime (default: 00:30:00)
  --predict-cpus <n>           Predict cpus-per-task (default: 32)
  --predict-mem <mem>          Predict memory (default: 256G)
  --predict-gpus <n>           Predict gpus-per-node (default: 1)
  --eval-qos <qos>             Slurm QoS for eval job (default: nf)
  --eval-time <hh:mm:ss>       Eval walltime (default: 08:00:00)
  --eval-cpus <n>              Eval cpus-per-task (default: 8)
  --eval-mem <mem>             Eval memory (default: 64G)
  --allow-existing-run-dir     Allow reusing an existing run directory (explicitly unsafe)
  --dry-run                    Generate scripts only
EOF
}

RUN_ID=""
EVAL_ROOT="/home/ecm5702/scratch/eval"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"
CKPT_ID="4a5b2f1b24b84c52872bfcec1410b00f"
NAME_CKPT=""
PREDICT_QOS="dg"
PREDICT_TIME="00:30:00"
PREDICT_CPUS="32"
PREDICT_MEM="256G"
PREDICT_GPUS="1"
EVAL_QOS="nf"
EVAL_TIME="08:00:00"
EVAL_CPUS="8"
EVAL_MEM="64G"
ALLOW_EXISTING_RUN_DIR=0
DRY_RUN=0

to_seconds() {
  local t="$1"
  local h=0 m=0 s=0
  IFS=":" read -r h m s <<<"$t"
  if [[ -z "$s" ]]; then
    s="$m"
    m="$h"
    h=0
  fi
  echo $((10#$h*3600 + 10#$m*60 + 10#$s))
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --ckpt-id) CKPT_ID="$2"; shift 2 ;;
    --name-ckpt) NAME_CKPT="$2"; shift 2 ;;
    --predict-qos) PREDICT_QOS="$2"; shift 2 ;;
    --predict-time) PREDICT_TIME="$2"; shift 2 ;;
    --predict-cpus) PREDICT_CPUS="$2"; shift 2 ;;
    --predict-mem) PREDICT_MEM="$2"; shift 2 ;;
    --predict-gpus) PREDICT_GPUS="$2"; shift 2 ;;
    --eval-qos) EVAL_QOS="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --eval-cpus) EVAL_CPUS="$2"; shift 2 ;;
    --eval-mem) EVAL_MEM="$2"; shift 2 ;;
    --allow-existing-run-dir) ALLOW_EXISTING_RUN_DIR=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
 done

if [[ -z "$RUN_ID" ]]; then
  echo "--run-id is required" >&2
  usage
  exit 2
fi
if [[ "${RUN_ID}" == *"/"* ]]; then
  echo "RUN_ID must not contain '/': ${RUN_ID}" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsafe characters: ${RUN_ID}" >&2
  echo "Allowed pattern: [A-Za-z0-9._-]+" >&2
  exit 2
fi

dg_max_sec=$((30*60))
ng_max_sec=$((48*3600))
req_sec="$(to_seconds "$PREDICT_TIME")"
if (( req_sec > dg_max_sec )) && [[ "$PREDICT_QOS" == "dg" ]]; then
  echo "PREDICT_TIME=$PREDICT_TIME exceeds dg max (00:30:00). Switching PREDICT_QOS to ng." >&2
  PREDICT_QOS="ng"
fi
if (( req_sec > ng_max_sec )); then
  echo "PREDICT_TIME=$PREDICT_TIME exceeds ng max (48:00:00). Aborting." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_ROOT_REAL="$(realpath -m "${EVAL_ROOT}")"
if [[ ( -d "${EVAL_ROOT}/logs" && -d "${EVAL_ROOT}/jobs" ) || ( -d "${EVAL_ROOT_REAL}/logs" && -d "${EVAL_ROOT_REAL}/jobs" ) ]]; then
  echo "Refusing eval root that already looks like a run directory: ${EVAL_ROOT}" >&2
  echo "Use the parent eval root (for example /home/ecm5702/scratch/eval), not an existing run folder." >&2
  exit 2
fi
RUN_DIR="${EVAL_ROOT}/${RUN_ID}"
JOBS_DIR="${RUN_DIR}/jobs"
PRED_DIR="${RUN_DIR}/predictions"
EVAL_RUN_ROOT="${RUN_DIR}/eval"

if [[ -e "${RUN_DIR}" && "${ALLOW_EXISTING_RUN_DIR}" -ne 1 ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Refusing silent reuse. Use a new run-id, or pass --allow-existing-run-dir explicitly." >&2
  exit 2
fi
mkdir -p "${JOBS_DIR}" "${RUN_DIR}/logs" "${PRED_DIR}" "${EVAL_RUN_ROOT}"

PREDICTION_SAFETY_FLAGS=""
if [[ "${ALLOW_EXISTING_RUN_DIR}" -eq 1 ]]; then
  PREDICTION_SAFETY_FLAGS="--allow-existing-out-dir --allow-overwrite-existing-files"
fi

CKPT_FLAG=""
if [[ -n "$NAME_CKPT" ]]; then
  CKPT_FLAG="--name-ckpt ${NAME_CKPT}"
else
  CKPT_FLAG="--ckpt-id ${CKPT_ID}"
fi

cat > "${JOBS_DIR}/predict25_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=p25_${RUN_ID}
#SBATCH --qos=${PREDICT_QOS}
#SBATCH --time=${PREDICT_TIME}
#SBATCH --cpus-per-task=${PREDICT_CPUS}
#SBATCH --gpus-per-node=${PREDICT_GPUS}
#SBATCH --mem=${PREDICT_MEM}
#SBATCH --output=${RUN_DIR}/logs/predict25_${RUN_ID}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction
cd ${PROJECT_ROOT}
python eval/jobs/generate_predictions_25_files.py \
  --input-root ${INPUT_ROOT} \
  --out-dir ${PRED_DIR} \
  ${CKPT_FLAG} \
  --device cuda ${PREDICTION_SAFETY_FLAGS}
EOF

cat > "${JOBS_DIR}/eval25_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=e25_${RUN_ID}
#SBATCH --qos=${EVAL_QOS}
#SBATCH --time=${EVAL_TIME}
#SBATCH --cpus-per-task=${EVAL_CPUS}
#SBATCH --mem=${EVAL_MEM}
#SBATCH --output=${RUN_DIR}/logs/eval25_${RUN_ID}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
cd ${PROJECT_ROOT}
count=0
for f in ${PRED_DIR}/predictions_*.nc; do
  [ -f "\$f" ] || continue
  base=\$(basename "\$f" .nc)
  python -m eval.run --eval-root ${EVAL_RUN_ROOT} predictions --predictions-nc "\$f" --run-name "\${base}" --skip-region
  count=\$((count+1))
  echo "evaluated \$count file(s): \$f"
done
if [ "\$count" -ne 25 ]; then
  echo "Expected 25 prediction files but evaluated \$count" >&2
  exit 3
fi
EOF

chmod +x "${JOBS_DIR}"/*.sbatch

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run. Scripts written to ${JOBS_DIR}"
  exit 0
fi

jid_predict=$(sbatch "${JOBS_DIR}/predict25_${RUN_ID}.sbatch" | awk '{print $NF}')
jid_eval=$(sbatch --dependency=afterok:${jid_predict} "${JOBS_DIR}/eval25_${RUN_ID}.sbatch" | awk '{print $NF}')

echo "Submitted:"
echo "  predict25 ${jid_predict}"
echo "  eval25    ${jid_eval} (afterok:${jid_predict})"
echo "Monitor: squeue -j ${jid_predict},${jid_eval}"
