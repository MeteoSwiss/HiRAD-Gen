#!/bin/bash
#SBATCH --job-name=o1280-regions
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/home/ecm5702/dev/outputs/region_plotting/o1280-regions-%j.out

set -euxo pipefail

source /home/ecm5702/dev/.ds-dyn/bin/activate
PROJECT_ROOT="/home/ecm5702/dev/downscaling-tools"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python - << 'PY'
from eval._backends.region_plotting.plot_regions import run_region_plots_from_predictions

run_region_plots_from_predictions(
    run_parent_dir="/home/ecm5702/scratch/eval",
    run_name="o1280_3c54794f_agstyle_evalcheck_full_20260227",
    predictions_filename="predictions.nc",
)
print("done")
PY
