#!/bin/bash
# Serve an mlflow sqlite db from inside the training container (so the mlflow version
# matches whatever wrote the db - a mismatched local mlflow client/server errors on
# schema version). Long-running, lightweight (no GPU).
#
# Usage: sbatch src/hirad/mlflow_server.sh <path-to-mlflow.db> [port]

#SBATCH --job-name="mlflow-server"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00

### OUTPUT ###
#SBATCH --output=./logs/mlflow_server.log
#SBATCH --error=./logs/mlflow_server.err

### ENVIRONMENT ####
#SBATCH -A c38

MLFLOW_DB_PATH="$(realpath "${1:?Usage: sbatch mlflow_server.sh /users/mmcgloho/hirad-gen/HiRAD-Gen/outputs/training/regression_ifso1280_real/mlflow.db
 [port]}")"
PORT="${2:-5551}"

echo "Node: $(hostname)"
echo "Serving $MLFLOW_DB_PATH on port $PORT"

srun --environment=./ci/edf/modulus_env.toml bash -c "
    mlflow server --backend-store-uri sqlite:///$MLFLOW_DB_PATH --host 0.0.0.0 --port $PORT
"
