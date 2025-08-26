#!/bin/bash -l
#
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --partition=xfer

echo -e "$SLURM_JOB_NAME started on $(date):\n $command $1 $2"
cp -rvn $1 $2

echo -e "$SLURM_JOB_NAME finished on $(date)\n"

