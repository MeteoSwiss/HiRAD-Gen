#!/bin/bash
# Fetch GRIB data from MARS.

#SBATCH --job-name="fetch-ifs-data"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=./logs/fetch_ifs_data_%j.log

### ENVIRONMENT ####
#SBATCH -A a0269

set -euo pipefail

source .venv/bin/activate
python src/hirad/input_data/ifs-hres/fetch_mars_grib.py