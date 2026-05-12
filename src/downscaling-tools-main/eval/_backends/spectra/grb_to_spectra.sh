#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/spectra_compute/compute_spectra-%j.out



set -e 
module unload ecmwf-toolbox
module load ifs

expid="j1eg"
thegrid=o320
thedate=20230801/to/20230810/by/1
thestep=144
conda_env="/home/ecm5702/dev/.ds-dyn/bin/activate"


### FIRST ARCHIVE OLD RESULTS IF NEEDED ###
archive_dir_if_nonempty () {
    local src="$1"
    local base_archive="$2"

    if [ -d "$src" ] && [ "$(ls -A "$src")" ]; then
        ts=$(date +%Y%m%d_%H%M%S)
        archived_dir="${base_archive}_${ts}"

        echo "[INFO] Archiving $src to $archived_dir"
        mv "$src" "$archived_dir"

        mkdir -p "$src"
    fi
}

base="$PERM/ai_spectra/$expid"

# spectral_harmonics
archive_dir_if_nonempty \
    "$base/spectral_harmonics" \
    "$base/old_spectral_harmonics"

# spectra
archive_dir_if_nonempty \
    "$base/spectra" \
    "$base/old_spectra"

# grb
archive_dir_if_nonempty \
    "$base/grb" \
    "$base/old_grb"



### FIRST STEP - LOAD GRIB FILES FROM MARS ###
echo "=== FIRST STEP - DOWNLOAD GRIB ==="
thetime=00
expclass=rd
outdir=/home/ecm5702/perm/ai_spectra/$expid/grb
mkdir -p $outdir
cd $outdir
echo $expid

params=(
  "2t:sfc"
  "10u:sfc"
  "10v:sfc"
  "sp:sfc"
  "t:850"
  "z:500"
)

for param_level in "${params[@]}"; do
  param="${param_level%%:*}"
  level="${param_level##*:}"
  
  mkdir -p "$outdir/${param}_${level}"
  
  if [ "$level" == "sfc" ]; then
    levtype="sfc"
    mars<<EOF
retrieve,
type=pf,
date=$thedate,
levtype=$levtype,
param=$param,
step=$thestep,
time=$thetime,
number=1/2,
grid=$thegrid,
stream=enfo,
class=$expclass,
expver=$expid,
area=89/0/-89/360,
target="$outdir/${param}_${level}/[expver]_[date]_[step]_[number]_nopoles.grb"
EOF
  else
    levtype="pl"
    mars<<EOF
retrieve,
type=pf,
date=$thedate,
levtype=$levtype,
level=$level,
param=$param,
step=$thestep,
time=$thetime,
number=1/2,
grid=$thegrid,
stream=enfo,
class=$expclass,
expver=$expid,
area=89/0/-89/360,
target="$outdir/${param}_${level}/[expver]_[date]_[step]_[number]_nopoles.grb"
EOF
  fi
done


### SECOND STEP - CONVERT GRIB FILES TO SPECTRAL HARMONICS ###
set -ex
echo "=== SECOND STEP - CONVERT GRIB FILES TO SPECTRAL HARMONICS ==="

module unload ecmwf-toolbox
module load eclib
module load pifsenv
module load ifs
# needed for gptosp.ser
export DR_HOOK_ASSERT_MPI_INITIALIZED=0




gptosp_extra_arg=""

indir=/home/ecm5702/perm/ai_spectra/$expid/grb
outdir=$PERM/ai_spectra/$expid/spectral_harmonics

# If spectral_harmonics exists and is not empty, archive it
if [ -d "$outdir" ] && [ "$(ls -A "$outdir")" ]; then
    ts=$(date +%Y%m%d_%H%M%S)
    archived_dir="$PERM/ai_spectra/$expid/old_spectral_harmonics_$ts"

    echo "[INFO] Archiving existing spectral_harmonics to $archived_dir"
    mv "$outdir" "$archived_dir"

    # Recreate empty spectral_harmonics directory
    mkdir -p "$outdir"
fi

wdir=${outdir}
mkdir -p $wdir
cd $wdir


# Define parameter directories to process
param_dirs=(
  "2t_sfc"
  "10u_sfc"
  "10v_sfc"
  "sp_sfc"
  "t_850"
  "z_500"
)
# Parse date range: 20230801/to/20230802/by/1
d_start=$(echo "$thedate" | cut -d'/' -f1)
d_end=$(echo   "$thedate" | cut -d'/' -f3)
d_inc=$(echo   "$thedate" | cut -d'/' -f5)
echo "[INFO] Date range: start=${d_start}, end=${d_end}, inc=${d_inc} day(s)"


# Process each parameter directory
for param_dir in "${param_dirs[@]}"; do
  echo "Processing $param_dir files..."
  mkdir -p "$wdir/$param_dir"

  d="$d_start"
  # Loop over dates
  while [ "$d" -le "$d_end" ]; do

    # Assuming filenames like: irpw_20230801_24_*.grb
    for xfile in ${indir}/${param_dir}/${expid}_${d}_*.grb; do
      [ -f "$xfile" ] || continue
      echo "Converting $xfile"
      basename_file=$(basename "$xfile")
      gptosp.ser -l $gptosp_extra_arg -g "$xfile" -S "$wdir/$param_dir/${basename_file}_sh"
    done


    # increment date by d_inc days
    d=$(date -d "${d} + ${d_inc} day" +%Y%m%d)
  done
done

echo "All files processed"
ls -la $wdir/*

### THIRD STEP - COMPUTE SPECTRA AMPLITUDES FROM SPECTRAL HARMONICS ###
set -ex
source $conda_env

echo "=== THIRD STEP - COMPUTE SPECTRA AMPLITUDES FROM SPECTRAL HARMONICS ==="

module unload ifs

module load ecmwf-toolbox

WORKDIR=/home/ecm5702/dev/jobscripts/outputs/spectra_compute
cd $WORKDIR


# Run inference for each experiment

python /home/ecm5702/dev/post_prepml/spectra/spectra_ml/individual_files/compute_spectra-3.py --expver $expid