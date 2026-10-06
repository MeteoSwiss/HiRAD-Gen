# IFS-HRES → RealCH1 input-data pipeline

Reusable scripts for turning raw IFS-HRES forecast GRIB into a training-ready anemoi
trajectories (5D) dataset, plus the normalization stats. All jobs run in the container
(`ci/edf/modulus_env.toml`) via SLURM; put `src` on `PYTHONPATH` instead of `pip install -e .`.

Only the **00Z/12Z** IFS-HRES runs carry the full 3D fields (pressure levels + hybrid q); the
06Z/18Z runs are surface-only and are skipped.

## 1. Create the input-channel GRIBs (per (init, step))

`ifs_hres_precompute.py` — builds the 13 input channels for one or more init cycles:
`2t,10u,10v,tcw,t/z/u/v_{850,500},tp`. `tcw` is derived from hybrid `q` + `sp`; `tp` is a 1h
accumulation `tp(L)-tp(L-1)`; derived fields encoded with low-level eccodes. Parallel over leads
with `--workers`; accepts many inits in one process (`--init-dirs`) to amortize imports.

`ifs_hres_precompute_full.sh` — SLURM array (one task per month) over a date range. Override
range/paths via env and set `--array` to the number of months − 1:

```bash
sbatch --export=ALL,START=2020-10-01,END=2025-02-28,OUT_DIR=/scratch/.../precompute,LEADS=1-33 \
       --array=0-52%16 src/hirad/input_data/ifs_hres_precompute_full.sh
```

Check completeness before building (fills the recipe's `base_dates.missing`):

```bash
python -m hirad.input_data.check_precompute_completeness \
    --out-dir /scratch/.../precompute --start 2020-10-01 --end 2025-02-28 --leads 1-33
```

## 2. Build the anemoi trajectories zarr

`anemoi_sources.py` registers the custom `ifs-hres-files` source (serves the precomputed GRIBs
to a forecast/trajectory build; the built-in `grib` source can't). `build_ifs_hres_anemoi.py`
runs the anemoi `create` tasks in-process (and monkeypatches a 0.5.41 timedelta bug).

Recipe template: `configs/ifs_hres_train_full.yaml` (edit `base_dates`, `steps`, `input.directory`,
`statistics`, and `base_dates.missing` from step 1).

```bash
sbatch --export=ALL,RECIPE=src/hirad/input_data/configs/ifs_hres_train_full.yaml,ZARR=/scratch/.../ds.zarr \
       src/hirad/input_data/build_ifs_hres_full.sh
```

## 3. Calculate normalization stats (tp Box-Cox)

The 12 non-precip channels use the zarr's built-in statistics. `tp` uses Box-Cox(0.25) mean/std,
computed on a seasonally-spread subset (`--stride`) so it isn't biased to one time of year:

```bash
sbatch --export=ALL,ZARR=/scratch/.../ds.zarr,STRIDE=8 \
       src/hirad/input_data/calculate_tp_boxcox_stats.sh
```

Paste the printed `transform_input_means/stdevs` into the dataset config
(`conf/dataset/anemoi_ifsn320_real_train.yaml`). The output tp constants stay as the RealCH1
`TOT_PREC_1H` values (same target as era-real).

## Consume in training

`conf/dataset/anemoi_ifsn320_real_train.yaml` points `input_anemoi_dataset_path` at the zarr,
sets `provide_lead_time: true`, and selects train/val by init-time `start_date/end_date` and
`validation_start_date/validation_end_date` (both filter base times; short train→val leakage via
long leads is accepted by design).
