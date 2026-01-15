#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=xfer

rsync -av /iopsstor/scratch/cscs/mmcgloho/basic-numpy /capstor/store/cscs/pasc/c38/basic-numpy