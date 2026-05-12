#!/usr/bin/env bash
set -euo pipefail

# ── Proxy TC Eval ─────────────────────────────────────────────────────────────
# Runs the 10 canonical date/step pairs with a configurable member subset.
# Default proxy member scope is member 1 only, so the fast gate writes
# 10 total date-step-member predictions while preserving the canonical pair set.
# That subset captures ~64% of the combined TC extreme signal from the full
# 250-run evaluation.
#
# Top-10 bundles (ranked by combined Idalia+Franklin MSLP/wind extremes):
#   20230829:24, 20230828:48, 20230829:48, 20230828:24, 20230830:24,
#   20230828:72, 20230827:72, 20230830:48, 20230829:72, 20230827:48
#
# Usage:
#   ./launch_proxy_eval.sh --run-id proxy_test1 --ckpt-id <checkpoint_id>
#   ./launch_proxy_eval.sh --run-id proxy_test1 --ckpt-id <checkpoint_id> --dry-run
# ──────────────────────────────────────────────────────────────────────────────

PROXY_BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"
PROXY_N_FILES=""
SCOREBOARD_SUBSET_DIR_NAME="proxy10total_subset"
SCOREBOARD_SPECTRA_DIR_NAME="spectra_harmonized_proxy10total"
SCOREBOARD_TC_EVENTS="idalia,franklin"
SCOREBOARD_TC_SUPPORT_MODE="regridded"
SCOREBOARD_BASE_TC_DIR="/home/ecm5702/perm/reference/o96_o320/tc"
SCOREBOARD_TC_DISPLAY_LABEL=""
SCOREBOARD_SPECTRA_WEATHER_STATES="10u,10v,2t,msl,t_850,z_500"
SCOREBOARD_SPECTRA_NSIDE=128
SCOREBOARD_SPECTRA_LMAX=319
SCOREBOARD_SPECTRA_MEMBER_AGG="per-file-mean"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --run-id <id> --ckpt-id <id> [options]

Proxy TC quick-eval: configurable proxy DATE:STEP pairs (default 10 files).
Default proxy member scope: member 1 only (default 10 total predictions under the canonical proxy10 contract, target ~30 min GPU, qos=dg).

Options:
  --run-id <id>               Required run id (e.g. proxy_v1)
  --eval-root <path>          Eval root (default: /home/ecm5702/scratch/eval)
  --input-root <path>         Bundle input root
  --proxy-bundle-pairs <csv>  Explicit proxy DATE:STEP pairs
  --proxy-n-files <n>         Expected proxy prediction/eval file count
  --ckpt-id <id>              Checkpoint id (required)
  --name-ckpt <path>          Explicit checkpoint path (overrides --ckpt-id)
  --predict-qos <qos>         Predict job QoS (default: dg)
  --predict-time <hh:mm:ss>   Predict walltime (default: 00:30:00)
  --predict-cpus <n>          Predict cpus-per-task (default: 32)
  --predict-mem <mem>         Predict memory (default: 256G)
  --predict-gpus <n>          Predict gpus-per-node (default: 1)
  --members <csv>             Proxy member ids (default: 1)
  --extra-args-json <json>    Optional sampler override JSON passed to prediction generation
  --eval-qos <qos>            Eval job QoS (default: nf)
  --eval-time <hh:mm:ss>      Eval walltime (default: 04:00:00)
  --eval-cpus <n>             Eval cpus-per-task (default: 8)
  --eval-mem <mem>            Eval memory (default: 64G)
  --write-scoreboard-artifacts
                              Also write strict proxy TC JSON/PDF + spectra summary
  --scoreboard-subset-dir-name <name>
                              Override proxy strict-artifact subset directory name
  --scoreboard-spectra-dir-name <name>
                              Override proxy strict-artifact spectra directory name
  --scoreboard-tc-events <csv>
                              Override TC event list used for strict artifacts
  --scoreboard-tc-support-mode <mode>
                              TC support mode for strict artifacts (native|regridded)
  --scoreboard-base-tc-dir <path>
                              Base TC directory for strict artifacts
  --scoreboard-tc-display-label <label>
                              Short TC PDF legend label; blank auto-cuts from run id
  --scoreboard-spectra-weather-states <csv>
                              Weather states for strict spectra generation
  --scoreboard-spectra-nside <n>
                              NSIDE for strict proxy spectra generation
  --scoreboard-spectra-lmax <n>
                              LMAX for strict proxy spectra generation
  --scoreboard-spectra-member-agg <mode>
                              Member aggregation for strict proxy spectra
  --allow-existing-run-dir    Allow reusing an existing run directory
  --skip-eval                 Only run predictions, skip evaluation
  --dry-run                   Generate scripts only
EOF
}

RUN_ID=""
EVAL_ROOT="/home/ecm5702/scratch/eval"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"
CKPT_ID=""
NAME_CKPT=""
PREDICT_QOS="dg"
PREDICT_TIME="00:30:00"
PREDICT_CPUS="32"
PREDICT_MEM="256G"
PREDICT_GPUS="1"
PREDICT_MEMBERS="1"
EXTRA_ARGS_JSON=""
EVAL_QOS="nf"
EVAL_TIME="04:00:00"
EVAL_CPUS="8"
EVAL_MEM="64G"
WRITE_SCOREBOARD_ARTIFACTS=0
ALLOW_EXISTING_RUN_DIR=0
SKIP_EVAL=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --input-root) INPUT_ROOT="$2"; shift 2 ;;
    --proxy-bundle-pairs) PROXY_BUNDLE_PAIRS="$2"; shift 2 ;;
    --proxy-n-files) PROXY_N_FILES="$2"; shift 2 ;;
    --ckpt-id) CKPT_ID="$2"; shift 2 ;;
    --name-ckpt) NAME_CKPT="$2"; shift 2 ;;
    --predict-qos) PREDICT_QOS="$2"; shift 2 ;;
    --predict-time) PREDICT_TIME="$2"; shift 2 ;;
    --predict-cpus) PREDICT_CPUS="$2"; shift 2 ;;
    --predict-mem) PREDICT_MEM="$2"; shift 2 ;;
    --predict-gpus) PREDICT_GPUS="$2"; shift 2 ;;
    --members) PREDICT_MEMBERS="$2"; shift 2 ;;
    --extra-args-json) EXTRA_ARGS_JSON="$2"; shift 2 ;;
    --eval-qos) EVAL_QOS="$2"; shift 2 ;;
    --eval-time) EVAL_TIME="$2"; shift 2 ;;
    --eval-cpus) EVAL_CPUS="$2"; shift 2 ;;
    --eval-mem) EVAL_MEM="$2"; shift 2 ;;
    --write-scoreboard-artifacts) WRITE_SCOREBOARD_ARTIFACTS=1; shift ;;
    --scoreboard-subset-dir-name) SCOREBOARD_SUBSET_DIR_NAME="$2"; shift 2 ;;
    --scoreboard-spectra-dir-name) SCOREBOARD_SPECTRA_DIR_NAME="$2"; shift 2 ;;
    --scoreboard-tc-events) SCOREBOARD_TC_EVENTS="$2"; shift 2 ;;
    --scoreboard-tc-support-mode) SCOREBOARD_TC_SUPPORT_MODE="$2"; shift 2 ;;
    --scoreboard-base-tc-dir) SCOREBOARD_BASE_TC_DIR="$2"; shift 2 ;;
    --scoreboard-tc-display-label) SCOREBOARD_TC_DISPLAY_LABEL="$2"; shift 2 ;;
    --scoreboard-spectra-weather-states) SCOREBOARD_SPECTRA_WEATHER_STATES="$2"; shift 2 ;;
    --scoreboard-spectra-nside) SCOREBOARD_SPECTRA_NSIDE="$2"; shift 2 ;;
    --scoreboard-spectra-lmax) SCOREBOARD_SPECTRA_LMAX="$2"; shift 2 ;;
    --scoreboard-spectra-member-agg) SCOREBOARD_SPECTRA_MEMBER_AGG="$2"; shift 2 ;;
    --allow-existing-run-dir) ALLOW_EXISTING_RUN_DIR=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then echo "--run-id is required" >&2; usage; exit 2; fi
if [[ -z "$CKPT_ID" && -z "$NAME_CKPT" ]]; then echo "--ckpt-id or --name-ckpt is required" >&2; usage; exit 2; fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID contains unsafe characters: ${RUN_ID}" >&2; exit 2
fi
IFS=',' read -r -a PROXY_PAIR_LIST <<< "${PROXY_BUNDLE_PAIRS}"
PROXY_PAIR_COUNT=0
for raw_pair in "${PROXY_PAIR_LIST[@]}"; do
  trimmed="${raw_pair//[[:space:]]/}"
  if [[ -n "${trimmed}" ]]; then
    PROXY_PAIR_COUNT=$((PROXY_PAIR_COUNT + 1))
  fi
done
if [[ "${PROXY_PAIR_COUNT}" -eq 0 ]]; then
  echo "--proxy-bundle-pairs must include at least one DATE:STEP pair" >&2
  exit 2
fi
if [[ -z "${PROXY_N_FILES}" ]]; then
  PROXY_N_FILES="${PROXY_PAIR_COUNT}"
fi
if [[ ! "${PROXY_N_FILES}" =~ ^[0-9]+$ ]] || [[ "${PROXY_N_FILES}" -le 0 ]]; then
  echo "--proxy-n-files must be a positive integer" >&2
  exit 2
fi
if [[ "${PROXY_N_FILES}" -ne "${PROXY_PAIR_COUNT}" ]]; then
  echo "--proxy-n-files=${PROXY_N_FILES} does not match the ${PROXY_PAIR_COUNT} configured DATE:STEP pairs" >&2
  exit 2
fi
IFS=',' read -r -a PROXY_MEMBER_LIST <<< "${PREDICT_MEMBERS}"
PROXY_MEMBER_COUNT=0
for raw_member in "${PROXY_MEMBER_LIST[@]}"; do
  trimmed="${raw_member//[[:space:]]/}"
  if [[ -n "${trimmed}" ]]; then
    PROXY_MEMBER_COUNT=$((PROXY_MEMBER_COUNT + 1))
  fi
done
if [[ "${PROXY_MEMBER_COUNT}" -eq 0 ]]; then
  echo "--members must include at least one member id" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCOREBOARD_SPECTRA_SCRIPT="${PROJECT_ROOT}/eval/jobs/templates/predictions_dir_spectra.py"

if [[ "${WRITE_SCOREBOARD_ARTIFACTS}" -eq 1 && ! -f "${SCOREBOARD_SPECTRA_SCRIPT}" ]]; then
  echo "Missing spectra helper: ${SCOREBOARD_SPECTRA_SCRIPT}" >&2
  exit 2
fi
RUN_DIR="${EVAL_ROOT}/${RUN_ID}"
JOBS_DIR="${RUN_DIR}/jobs"
PRED_DIR="${RUN_DIR}/predictions"
EVAL_RUN_ROOT="${RUN_DIR}/eval"
EXTRA_ARGS_JSON_ESCAPED="$(printf '%q' "${EXTRA_ARGS_JSON}")"

if [[ -e "${RUN_DIR}" && "${ALLOW_EXISTING_RUN_DIR}" -ne 1 ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  echo "Use a new run-id, or pass --allow-existing-run-dir explicitly." >&2
  exit 2
fi
mkdir -p "${JOBS_DIR}" "${RUN_DIR}/logs" "${PRED_DIR}" "${EVAL_RUN_ROOT}"

PYTHON_FOR_CONFIG="${PYTHON_FOR_CONFIG:-/home/ecm5702/dev/.ds-dyn/bin/python}"
if [[ ! -x "${PYTHON_FOR_CONFIG}" ]]; then
  PYTHON_FOR_CONFIG="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_FOR_CONFIG}" ]]; then
  echo "No Python interpreter found for writing EXPERIMENT_CONFIG.yaml" >&2
  exit 2
fi

"${PYTHON_FOR_CONFIG}" - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${RUN_ID}" "${RUN_DIR}" "${CKPT_ID}" "${NAME_CKPT}" "${INPUT_ROOT}" "${PROXY_BUNDLE_PAIRS}" "${PREDICT_MEMBERS}" "${EXTRA_ARGS_JSON}" <<'PY'
import json
import sys
from pathlib import Path


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


out_path = Path(sys.argv[1])
run_id = sys.argv[2]
run_dir = sys.argv[3]
ckpt_id = sys.argv[4].strip()
name_ckpt = sys.argv[5].strip()
input_root = sys.argv[6]
bundle_pairs = sys.argv[7]
members = parse_int_list(sys.argv[8])
sampling_config_json = sys.argv[9]

dates: set[int] = set()
steps: set[int] = set()
for pair in bundle_pairs.split(","):
    pair = pair.strip()
    if not pair:
        continue
    date_text, step_text = pair.split(":", 1)
    dates.add(int(date_text))
    steps.add(int(step_text))

config = {
    "run_id": run_id,
    "artifacts_root": run_dir,
    "sampling_config_json": sampling_config_json,
    "checkpoint": {
        "run_id": ckpt_id or (Path(name_ckpt).parts[0] if name_ckpt else "na"),
        "path": name_ckpt or "na",
    },
    "source": {
        "input_root": input_root,
        "bundle_scope": {
            "dates": sorted(dates),
            "members": members,
            "steps_hours": sorted(steps),
        },
    },
}
if sampling_config_json:
    try:
        config["model"] = {"development_hacks": {"extra_args": json.loads(sampling_config_json)}}
    except json.JSONDecodeError:
        pass

out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

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

# ── predict proxy sbatch ──────────────────────────────────────────────────────
cat > "${JOBS_DIR}/predict_proxy_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=pxy_${RUN_ID}
#SBATCH --qos=${PREDICT_QOS}
#SBATCH --time=${PREDICT_TIME}
#SBATCH --cpus-per-task=${PREDICT_CPUS}
#SBATCH --gpus-per-node=${PREDICT_GPUS}
#SBATCH --mem=${PREDICT_MEM}
#SBATCH --output=${RUN_DIR}/logs/predict_proxy_${RUN_ID}_%j.out
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
EXTRA_ARGS_JSON=${EXTRA_ARGS_JSON_ESCAPED}
extra_args=()
if [[ -n "\${EXTRA_ARGS_JSON}" ]]; then
  extra_args+=(--extra-args-json "\${EXTRA_ARGS_JSON}")
fi
python eval/jobs/generate_predictions_25_files.py \\
  --input-root ${INPUT_ROOT} \\
  --out-dir ${PRED_DIR} \\
  ${CKPT_FLAG} \\
  --bundle-pairs "${PROXY_BUNDLE_PAIRS}" \\
  --members "${PREDICT_MEMBERS}" \\
  "\${extra_args[@]}" \\
  --device cuda ${PREDICTION_SAFETY_FLAGS}
EOF

# ── eval proxy sbatch ─────────────────────────────────────────────────────────
cat > "${JOBS_DIR}/eval_proxy_${RUN_ID}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=exy_${RUN_ID}
#SBATCH --qos=${EVAL_QOS}
#SBATCH --time=${EVAL_TIME}
#SBATCH --cpus-per-task=${EVAL_CPUS}
#SBATCH --mem=${EVAL_MEM}
#SBATCH --output=${RUN_DIR}/logs/eval_proxy_${RUN_ID}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
RUN_ID="${RUN_ID}"
RUN_DIR="${RUN_DIR}"
PRED_DIR="${PRED_DIR}"
WRITE_SCOREBOARD_ARTIFACTS=${WRITE_SCOREBOARD_ARTIFACTS}
SCOREBOARD_SUBSET_DIR_NAME="${SCOREBOARD_SUBSET_DIR_NAME}"
SCOREBOARD_SPECTRA_DIR_NAME="${SCOREBOARD_SPECTRA_DIR_NAME}"
SCOREBOARD_TC_EVENTS="${SCOREBOARD_TC_EVENTS}"
SCOREBOARD_TC_SUPPORT_MODE="${SCOREBOARD_TC_SUPPORT_MODE}"
SCOREBOARD_BASE_TC_DIR="${SCOREBOARD_BASE_TC_DIR}"
SCOREBOARD_TC_DISPLAY_LABEL="${SCOREBOARD_TC_DISPLAY_LABEL}"
SCOREBOARD_SPECTRA_SCRIPT="${SCOREBOARD_SPECTRA_SCRIPT}"
SCOREBOARD_SPECTRA_WEATHER_STATES="${SCOREBOARD_SPECTRA_WEATHER_STATES}"
SCOREBOARD_SPECTRA_NSIDE=${SCOREBOARD_SPECTRA_NSIDE}
SCOREBOARD_SPECTRA_LMAX=${SCOREBOARD_SPECTRA_LMAX}
SCOREBOARD_SPECTRA_MEMBER_AGG="${SCOREBOARD_SPECTRA_MEMBER_AGG}"
cd ${PROJECT_ROOT}
count=0
for f in \${PRED_DIR}/predictions_*.nc; do
  [ -f "\$f" ] || continue
  base=\$(basename "\$f" .nc)
  python -m eval.run --eval-root ${EVAL_RUN_ROOT} predictions --predictions-nc "\$f" --run-name "\${base}" --skip-region
  count=\$((count+1))
  echo "evaluated \$count file(s): \$f"
done
if [ "\$count" -ne ${PROXY_N_FILES} ]; then
  echo "Expected ${PROXY_N_FILES} prediction files but evaluated \$count" >&2
  exit 3
fi

# ── TC extreme comparison against anchor ──────────────────────────────────────
ANCHOR_JSON="${EVAL_ROOT}/anchor_tc_extremes.json"
if [ -f "\${ANCHOR_JSON}" ]; then
  echo "Running proxy TC comparison against anchor..."
  set +e
  python -m eval.jobs.proxy_tc_compare \\
    --proxy-predictions-dir ${PRED_DIR} \\
    --anchor-json "\${ANCHOR_JSON}" \\
    --out-json ${RUN_DIR}/proxy_tc_compare.json \\
    --support-mode native
  compare_rc=\$?
  set -e
  if [ ! -f "${RUN_DIR}/proxy_tc_compare.json" ]; then
    echo "proxy_tc_compare did not write ${RUN_DIR}/proxy_tc_compare.json" >&2
    exit "\${compare_rc}"
  fi
  echo "TC comparison saved to ${RUN_DIR}/proxy_tc_compare.json"
  if [ "\${compare_rc}" -ne 0 ]; then
    echo "Proxy TC comparison failed with exit code \${compare_rc}" >&2
    exit "\${compare_rc}"
  fi
else
  echo "No anchor TC extremes JSON at \${ANCHOR_JSON} — skipping TC comparison."
  echo "Generate one with: python -m eval.jobs.diagnose_per_bundle_tc_extremes --predictions-dir <anchor_predictions> --out-json \${ANCHOR_JSON} --analysis-expid <analysis_expid>"
fi

if [[ "\${WRITE_SCOREBOARD_ARTIFACTS}" -eq 1 ]]; then
  [[ -f "\${SCOREBOARD_SPECTRA_SCRIPT}" ]] || {
    echo "Missing spectra helper: \${SCOREBOARD_SPECTRA_SCRIPT}" >&2
    exit 4
  }
  SUBSET_DIR="\${RUN_DIR}/\${SCOREBOARD_SUBSET_DIR_NAME}"
  SUBSET_PRED_DIR="\${SUBSET_DIR}/predictions"
  SPECTRA_DIR="\${RUN_DIR}/\${SCOREBOARD_SPECTRA_DIR_NAME}"
  mkdir -p "\${SUBSET_PRED_DIR}" "\${SPECTRA_DIR}"
  for f in \${PRED_DIR}/predictions_*.nc; do
    [ -f "\$f" ] || continue
    ln -sfn "\$f" "\${SUBSET_PRED_DIR}/\$(basename "\$f")"
  done
  if [[ "\${SCOREBOARD_TC_SUPPORT_MODE}" == "regridded" ]]; then
    module load ecmwf-toolbox
    METVIEW_BIN="\$(command -v metview || true)"
    if [[ -n "\${METVIEW_BIN}" ]]; then
      export PATH="\$(dirname "\${METVIEW_BIN}"):\${PATH}"
    fi
    python -c "import metview" >/dev/null 2>&1
  fi
  TC_EVENT_TAG="\${SCOREBOARD_TC_EVENTS//,/_}"
  TC_EVENT_TAG="\${TC_EVENT_TAG// /}"
  python -m eval._backends.tc.workflows pdf \\
    --predictions-dir "\${SUBSET_PRED_DIR}" \\
    --outdir "\${SUBSET_DIR}" \\
    --run-label "\${RUN_ID}" \\
    --out-name "tc_normed_pdfs_\${TC_EVENT_TAG}_\${RUN_ID}_strict.pdf" \\
    --base-tc-dir "\${SCOREBOARD_BASE_TC_DIR}" \\
    --support-mode "\${SCOREBOARD_TC_SUPPORT_MODE}" \\
    --events "\${SCOREBOARD_TC_EVENTS}" \\
    --display-label "\${SCOREBOARD_TC_DISPLAY_LABEL:-\${RUN_ID}}"
  python "\${SCOREBOARD_SPECTRA_SCRIPT}" \\
    --predictions-dir "\${SUBSET_PRED_DIR}" \\
    --out-dir "\${SPECTRA_DIR}" \\
    --run-label "\${RUN_ID}" \\
    --weather-states "\${SCOREBOARD_SPECTRA_WEATHER_STATES}" \\
    --nside "\${SCOREBOARD_SPECTRA_NSIDE}" \\
    --lmax "\${SCOREBOARD_SPECTRA_LMAX}" \\
    --member-aggregation "\${SCOREBOARD_SPECTRA_MEMBER_AGG}"
  echo "Strict proxy scoreboard artifacts saved under \${SUBSET_DIR} and \${SPECTRA_DIR}"
fi
EOF

chmod +x "${JOBS_DIR}"/*.sbatch

echo "Proxy eval config:"
echo "  bundle-pairs: ${PROXY_BUNDLE_PAIRS}"
echo "  members: ${PREDICT_MEMBERS}"
echo "  n_files: ${PROXY_N_FILES}"
echo "  n_member_predictions: $((PROXY_N_FILES * PROXY_MEMBER_COUNT))"
echo "  tc_events: ${SCOREBOARD_TC_EVENTS}"
echo "  tc_display_label: ${SCOREBOARD_TC_DISPLAY_LABEL:-auto-cut from RUN_ID}"
echo "  spectra_weather_states: ${SCOREBOARD_SPECTRA_WEATHER_STATES}"
echo "  extra_args_json: ${EXTRA_ARGS_JSON:-<none>}"
echo "  scoreboard_artifacts: ${WRITE_SCOREBOARD_ARTIFACTS}"
echo "  predict qos: ${PREDICT_QOS}, time: ${PREDICT_TIME}"
echo "  scripts: ${JOBS_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run. Scripts written to ${JOBS_DIR}"
  exit 0
fi

jid_predict=$(sbatch "${JOBS_DIR}/predict_proxy_${RUN_ID}.sbatch" | awk '{print $NF}')

if [[ "${SKIP_EVAL}" -eq 1 ]]; then
  echo "Submitted predict_proxy: ${jid_predict} (eval skipped)"
  echo "Monitor: squeue -j ${jid_predict}"
else
  jid_eval=$(sbatch --dependency=afterok:${jid_predict} "${JOBS_DIR}/eval_proxy_${RUN_ID}.sbatch" | awk '{print $NF}')
  echo "Submitted:"
  echo "  predict_proxy ${jid_predict}"
  echo "  eval_proxy    ${jid_eval} (afterok:${jid_predict})"
  echo "Monitor: squeue -j ${jid_predict},${jid_eval}"
fi
