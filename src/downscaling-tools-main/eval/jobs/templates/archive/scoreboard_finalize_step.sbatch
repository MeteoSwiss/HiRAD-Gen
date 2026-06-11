#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
#SBATCH --job-name=scoreboard_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail
RUN_ID="REPLACE_RUN_ID"
OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_ROOT=""
PRED_DIR=""
EXPECTED_PRED_FILES=25
SIGMA_RUN_ID=""
SCOREBOARD_MD="/home/ecm5702/scratch/eval/scoreboards/enfo_o320_scoreboard.md"
WRITE_CLEAN_SCOREBOARDS=1
WRITE_METRICS_JSON=1
[[ "${RUN_ID}" != REPLACE_* ]] || { echo "ERROR: set RUN_ID" >&2; exit 2; }
[[ -n "${RUN_ROOT}" ]] || RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
REPO_ROOT="${DOWNSCALING_TOOLS_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
if [[ "${WRITE_CLEAN_SCOREBOARDS}" -eq 1 ]]; then
  python "${REPO_ROOT}/eval/jobs/generate_clean_scoreboards.py"
fi
python "${REPO_ROOT}/eval/jobs/generate_enfo_o320_scoreboard.py" --eval-root "${OUTPUT_ROOT}" --out-md "${SCOREBOARD_MD}"
if [[ "${WRITE_METRICS_JSON}" -eq 1 ]]; then
  python - "${RUN_ID}" "${OUTPUT_ROOT}" "${SIGMA_RUN_ID}" "${RUN_ROOT}" "${REPO_ROOT}" <<"PY"
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from eval.jobs import scoreboard_metrics as metrics
run_id, output_root, sigma_run_id, run_root = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out = metrics.build_run_scoreboard_metrics(
    run_id=run_id,
    output_root=output_root,
    sigma_run_id=sigma_run_id,
    tc_stats_path=run_root / "data" / "tc" / "idalia__franklin.stats.json",
    spectra_dir=run_root / "data" / "spectra",
    surface_json_path=run_root / "data" / "surface_loss.json",
)
(run_root / "metrics.json").write_text(json.dumps(out, indent=2))
PY
fi
echo "DONE scoreboard_finalize_step run_id=${RUN_ID}"
echo "SCOREBOARD_MD=${SCOREBOARD_MD}"
