#!/bin/bash
# Capture golden records, one array task per setup (stages run sequentially, one process each).
#
#   OUT=/capstor/.../golden/<label> SETUPS="era_real_cv1 era_cosmo" \
#     sbatch -A <account> --array=0-1 tests/golden/run_capture.sh
#
# SETUPS defaults to every setup in registry.yaml (then use --array=0-<n_setups - 1>);
# STAGES defaults to "generate train_regression train_diffusion"; GROUP_NORM to apex (torch: the
# bitwise-reproducible mode, compared exactly). Submit from the repo root.
#SBATCH --job-name=hirad-golden
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/golden_%A_%a.log

set -euo pipefail
: "${OUT:?set OUT to the capture root}"
STAGES="${STAGES:-generate train_regression train_diffusion}"
GROUP_NORM="${GROUP_NORM:-apex}"
if [[ -z "${SETUPS:-}" ]]; then
    SETUPS=$(sed -n '/^setups:/,$p' tests/golden/registry.yaml | grep -E '^  [a-z0-9_]+:$' | tr -d ' :' | tr '\n' ' ')
fi
read -r -a SETUP_LIST <<< "${SETUPS}"
SETUP="${SETUP_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "setup=${SETUP} stages=${STAGES} group_norm=${GROUP_NORM} out=${OUT}"

srun --environment=./ci/edf/modulus_env.toml bash -c "
    set -uo pipefail
    ulimit -c 0
    export PYTHONPATH=\${PWD}/src:\${PWD}:\${PYTHONPATH:-}
    status=0
    for stage in ${STAGES}; do
        python -m tests.golden.capture --setup ${SETUP} --stage \${stage} --group-norm ${GROUP_NORM} --out ${OUT} || status=1
    done
    exit \${status}
"
