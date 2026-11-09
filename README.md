# HiRAD-Gen

HiRAD-Gen — **Hi**gh-**R**esolution **A**tmospheric **D**ownscaling using **Gen**erative
models. It trains and runs CorrDiff-style generative models that downscale coarse atmospheric
fields (reanalysis or forecasts) to a high-resolution target grid. A typical setup downscales
ERA5 / IFS input (~10–30 km) to a 1-2 km target grid (REAL-CH1, COSMO), predicting surface variables
such as 2 m temperature (`2t`), 10 m wind (`10u`, `10v`) and total precipitation (`tp`).

This README explains, in order:

- [How it works](#how-it-works) — the two-stage model.
- [Data](#data) — how input and target datasets are expected to look and how they are paired.
- [Training](#training) — how to train the regression and diffusion stages.
- [Inference](#inference) — how to generate downscaled predictions.
- [Evaluation & visualization](#evaluation--visualization).
- [Configuration layout](#configuration-layout) — how the Hydra configs compose.
- [Extending HiRAD-Gen](#extending-hirad-gen-custom-datasets) — custom datasets, losses, models.
- [Running on CSCS Alps](#running-on-cscs-alps-claridensantis) — cluster/SLURM specifics.
- [MeteoSwiss-internal tools](#meteoswiss-internal-tools) — EvalML and internal data paths.

The first sections are platform-independent: they describe the codebase and its interfaces.
Everything specific to the CSCS Alps cluster (SLURM, the container image, accounts) lives in
its own section, and everything that needs MeteoSwiss-internal infrastructure (EvalML, internal
zarr stores, grid-info files, Balfrin) is collected at the very end so that external users can
skip it.

## Showcase

<table>
  <tr>
    <th width="5%"></th>
    <th width="31%">Input ERA5</th>
    <th width="31%">Prediction</th>
    <th width="31%">Target REAL-CH1</th>
  </tr>
  <tr>
    <td><b>2t</b></td>
    <td><img src="docs/images/showcase/2t-input.png" width="100%"></td>
    <td><img src="docs/images/showcase/2t-pred.png" width="100%"></td>
    <td><img src="docs/images/showcase/2t-target.png" width="100%"></td>
  </tr>
  <tr>
    <td><b>10u</b></td>
    <td><img src="docs/images/showcase/10-input.png" width="100%"></td>
    <td><img src="docs/images/showcase/10u-pred.png" width="100%"></td>
    <td><img src="docs/images/showcase/10-target.png" width="100%"></td>
  </tr>
  <tr>
    <td><b>10v</b></td>
    <td><img src="docs/images/showcase/10v-input.png" width="100%"></td>
    <td><img src="docs/images/showcase/10v-pred.png" width="100%"></td>
    <td><img src="docs/images/showcase/10v-target.png" width="100%"></td>
  </tr>
  <tr>
    <td><b>tp</b></td>
    <td><img src="docs/images/showcase/tp-input.png" width="100%"></td>
    <td><img src="docs/images/showcase/tp-pred1.png" width="100%"></td>
    <td><img src="docs/images/showcase/tp-target.png" width="100%"></td>
  </tr>
</table>

### Ensemble total precipitation (1 h)

<table>
  <tr>
    <td width="50%"><img src="docs/images/showcase/tp-pred1.png" width="100%"></td>
    <td width="50%"><img src="docs/images/showcase/tp-pred2.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="docs/images/showcase/tp-pred3.png" width="100%"></td>
    <td><img src="docs/images/showcase/tp-pred4.png" width="100%"></td>
  </tr>
</table>

## How it works

HiRAD-Gen is a two-stage **CorrDiff**-style downscaler:

1. **Regression stage (mean predictor).** A deterministic model maps the coarse input to a
   first estimate of the high-resolution field. It captures the conditional mean but is smooth.
2. **Diffusion stage (residual predictor).** A diffusion model learns the
   *residual* between the regression estimate and the true high-resolution field. Sampling it
   adds the fine-scale, physically plausible structure the regression stage cannot produce, and
   lets you draw an **ensemble** of equally likely high-resolution realizations.

The diffusion stage is trained *after* the regression stage and is conditioned on the regression
output (`hr_mean_conditioning`). Because the target grids are large, diffusion stage supports
**patch-based** training and sampling: the domain is split into overlapping patches so the model
sees a fixed patch size regardless of the full grid.

Optional conditioning signals, each toggled from config:

- **Lead-time conditioning** — for forecast inputs, the forecast lead time is optionally fed to the network
  so one model can learn different lead times biases.
- **Previous-HR (temporal) conditioning** — the diffusion stage can optionally condition on the
  previous timestep's high-resolution field for temporal consistency (autoregressive rollout at
  inference). See the `*_prevhr` configs.

The relevant source trees:

- `src/hirad/datasets/` - dataset classes (`base.py`, `anemoi_dataset.py`, `anemoi_forecast_dataset.py`, …).
- `src/hirad/models/` - `song_unet.py`, `unet.py`, `preconditioning.py`, `layers.py`.
- `src/hirad/losses/` - `loss.py`.
- `src/hirad/training/` - `train.py`, `training_manager.py`.
- `src/hirad/inference/` - main and helper generation scripts.
- `src/hirad/eval/` - various stand-alone evaluation scripts.
- `src/hirad/input_data/` - scripts to build/preprocess the input/target datasets from raw sources.
- `src/hirad/conf/` - Hydra configs (see [Configuration layout](#configuration-layout)).

## Data

HiRAD-Gen consumes [**anemoi-datasets**](https://anemoi.readthedocs.io/) `zarr` stores. A run
always pairs **two** stores:

- an **input** (coarse / low-resolution) store — the predictor fields, and
- a **target** (high-resolution) store — the fields the model learns to produce.

Both are opened with `anemoi.datasets.open_dataset`, which gives per-channel **statistics**
(mean/stdev) and the lat/lon geometry for free. You select the variables you want **by name**. You also select date ranges used for training and validation;
you do not have to pre-trim the stores to the exact channel set and date range.

### Two input layouts: analysis vs. forecast

The input store can have one of two time structures, and the dataset class is chosen by the
`type` string (`anemoi_<input>_<target>`):

| Input kind | `type` example | Dataset class | Store layout |
|------------|----------------|---------------|--------------|
| **Analysis / reanalysis** (one field per valid time) | `anemoi_era5_real`, `anemoi_era5_cosmo` | `AnemoiDataset` | 4D: `(dates, variables, ensemble, values)`, indexed by `.dates` |
| **Forecast / trajectory** ( one field per init time × lead) | `anemoi_ifs_real` | `AnemoiForecastDataset` | 5D: `(base_dates, variables, ensemble, step, values)`, indexed by `.base_dates` × `.steps` (anemoi-datasets ≥ 0.5.38) |

- **Analysis input.** Iteration is driven by the input store: every input valid time is paired
  with the target field at the **same valid time**. Input and target may have different native
  frequencies (e.g. 6-hourly input against an hourly target) — set `input_frequency` (in hours) in dataset config to resample
  the input, and only matching target times are used.
- **Forecast input.** One sample = one `(reference_time, step)` pair. Each forecast step is
  downscaled independently against the target field at its valid time (`reference_time + step`).
  Train/validation are split by **init time** (`start_date`/`end_date` filter the base dates).
  A `max_lead_hours` cap limits up to which lead times are used from input dataset.

> **Channel ordering.** For analysis stores the variables are selected with anemoi's `select=`,
> so the channel order is guaranteed to match `input_channel_names`. Forecast/trajectory stores
> cannot be opened with `select=` and have been observed to store variables **alphabetically**;
> `AnemoiForecastDataset` therefore maps names to native positions and reorders both the data and
> the statistics explicitly. If you add a new forecast store, double-check channel order — it has
> been a recurring source of silent bugs.

### What a dataset config contains

Datasets are configured under `src/hirad/conf/dataset/`. The fields (all consumed by
`AnemoiDataset.__init__` / `AnemoiForecastDataset.__init__`):

```yaml
type: anemoi_era5_real                 # <input>_<target>; selects the dataset class
input_anemoi_dataset_path:  /path/to/input.zarr    # coarse predictor store
target_anemoi_dataset_path: /path/to/target.zarr   # high-res target store

# Variables selected by name. Surface vars are bare; pressure-level vars are "<var>_<level>".
input_channel_names:  [2t, 10u, 10v, tcw, t_850, z_850, u_850, v_850, t_500, z_500, u_500, v_500, tp]
output_channel_names: [2t, 10u, 10v, tp]
static_channel_names: ['FIS']          # time-invariant fields (e.g. orography) from the target

# Per-channel transforms. tp is strongly non-Gaussian, so it is Box-Cox transformed before normalization; the transformed mean/stdev are supplied explicitly (precomputed, see below).
# Other channels and other transforms can be used, but dataset class has to be extended accordingly.
transform_channels:       ['tp-box_cox_025']
transform_input_means:    {'tp-box_cox_025': -3.8152}
transform_input_stdevs:   {'tp-box_cox_025':  0.2285}
transform_output_means:   {'tp-box_cox_025': -3.9236}
transform_output_stdevs:  {'tp-box_cox_025':  0.1926}

n_month_hour_channels: 4               # cyclic hour/month conditioning fed to the network as additional input
start_date: '2005-01-02'               # train range (init time for forecast inputs)
end_date:   '2020-12-31'
trim_edge: 41                          # pixels trimmed from each edge of the target

# Validation split
validation: True
validation_start_date: '2021-01-01'
validation_end_date:   '2021-12-31'

# Optional: carve a contiguous hold-out window out of [start_date, end_date]
# (e.g. a cross-validation fold). Analysis datasets only.
exclude_start_date: null
exclude_end_date:   null
```

Forecast datasets additionally take:

```yaml
provide_lead_time: True                
# feed lead time label to the network (lead-time conditioning)
max_lead_hours: 12                     
# use inputs up to this lead time, ignore ones above it
```

Key behaviors worth knowing:

- **Normalization.** By default each channel is normalized with the statistics stored in its own
  zarr. When the *inference* input comes from a **different** source than training (e.g. train on
  ERA5 analysis, run on IFS forecasts), you must normalize with the **training** stats so the
  model sees inputs in the distribution it was trained on. Set
  `input_stats_anemoi_dataset_path` (and, if needed, `target_stats_anemoi_dataset_path`) in dataset config to point
  at the training store. The `transform_*_means/stdevs` for Box-Cox channels must likewise match
  the checkpoint's training values.
- **Precipitation accumulation.** `tp` must be accumulated consistently for both input and target. If model is trained on 1h accumulation input and target data, during inference on new data it must also be fed 1h input accumulated data. Precip unit and
  accumulation handling is the single most common source of bugs here — treat it carefully.
- **Static fields** (e.g. `FIS` orography) are read once from the target store, normalized, and
  regridded to the target grid.
- **Missing targets.** For pure generation (no ground truth), set `target_missing_as_zeros: true`
  so iteration is driven by the input and a zeros target is emitted where the target store has no
  data.



## Training

You always train in **two stages — regression first, then
diffusion** — because the diffusion stage learns the residual on top of (and conditioned on) the
regression stage's output. Both stages share one entrypoint and are selected entirely by a
top-level Hydra config:

```bash
python src/hirad/training/train.py --config-name=<training_config>.yaml
```


### What a training config is made of

A top-level training config is almost entirely a `defaults:` list that composes one config from
each group, plus the Hydra output directory. For example
`src/hirad/conf/training_era_real_regression.yaml`:

```yaml
hydra:
  job:
    name: regression_era_real
  run:
    dir: /path/to/save/training/outputs    # checkpoints + logs land here in a subdirectory named as field hydra.job.name

defaults:
  - _self_
  - dataset/anemoi_era_real          # input/target stores, channels, dates  (see "Data")
  - model/era_real_regression        # model parameters
  - model_size/normal_real           # width/depth preset
  - training/era_real_regression     # optimizer, schedule, batch size, patching, model optimizations, I/O
  - logging/era_real_regression      # MLflow logging
```

So "how to train" is really "pick/compose the right config in each group". The groups:

**`dataset/`** — the input and target stores, channel names, transforms, and the train /
validation date ranges. This is where most of the information about *your* problem lives; see the
[Data](#data) section. The validation split is driven from here: set `validation: True` plus
`validation_start_date` / `validation_end_date` and the trainer builds a second dataset instance
for the validation range automatically.

**`model/`** — the network and its conditioning. Key fields:

```yaml
name: regression              # "regression" (stage 1) or "diffusion/patched_diffusion" (stage 2)
hr_mean_conditioning: False   # True for diffusion: condition on the regression mean
model_args:                   # architecture knobs passed to the U-Net, for all model_args see the UNet class and subclasses init methods
  N_grid_channels: 4          # positional grid-embedding channels (100 + gridtype: "learnable" for diffusion)
  embedding_type: zero        # timestep embedding: "zero" (regression), "positional"/"fourier" (diffusion)
# optional conditioning switches:
# lead_time_mode: True        # forecast lead-time conditioning (needs provide_lead_time set to True in dataset config)
# prev_hr_conditioning: True  # previous-HR temporal conditioning (needs provide_prev_target set to True in dataset config)
```

You do **not** set the network's input/output channel counts or grid size: they are
auto-derived at runtime from the dataset (`input_channels()`, `output_channels()`,
`static_channels()`, `image_shape()`, plus the cyclic time features and any lead-time/prev-HR
channels). Change the channel set in the dataset config and the model resizes itself. This is why
the same `model/` config works across datasets with different channel counts.

**`model_size/`** — width/depth preset (`model_channels`, `channel_mult`, `attn_resolutions`).
`normal_real` is ~80 M parameters, a good default for full-size grids.

**`training/`** — everything about the optimization loop, in three blocks:

```yaml
hp:                             # hyperparameters
  training_duration: 3000000    # stop after this many processed samples
  total_batch_size: "auto"      # global batch (auto = batch_size_per_gpu * world_size)
  batch_size_per_gpu: 4
  lr: 0.0002
  lr_rampup: 0                  # warmup, in samples
  lr_decay: 1                   # decay factor ...
  lr_decay_rate: 5e5            # ... applied every this many samples
  grad_clip_threshold: null
  # patch-based training (diffusion / large grids):
  # patch_shape_x: 384
  # patch_shape_y: 384          # patching is used when these are smaller than the grid
  # patch_num: 2                # patches drawn per sample (num_patches = batch_size*patch_num)
  # max_patch_per_gpu: 300      # if smaller than num_patches, inference is split and gradients are accumulated in iterations
perf:
  fp_optimizations: amp-bf16    # fp32 | fp16 | amp-fp16 | amp-bf16
  dataloader_workers: 10
  songunet_checkpoint_level: 0  # >0 = gradient checkpointing (trades compute for memory)
  use_apex_gn: True # optimized group norm from Apex
  torch_compile: True
io:
  print_progress_freq: 1000
  save_checkpoint_freq: 100000  # checkpoint cadence, in processed samples
  validation_freq: 50000
  validation_steps: 55          # steps for validation run, total_validated_samples = validation_steps * batch_size
  checkpoint_dir: .             # relative to the Hydra run dir
  # regression_checkpoint_path: /path/to/regression/ckpt   # REQUIRED for the diffusion stage
```

**`logging/`** — MLflow. Set `method: mlflow`, an `experiment_name`/`run_name`, and a `uri` for a
remote tracking server (omit/`null` to log locally in the run dir). See
[MLflow logging](#mlflow-logging).

### 1. Train the regression stage

```bash
python src/hirad/training/train.py --config-name=training_era_real_regression.yaml
```

Checkpoints and logs land in `hydra.run.dir` (under `checkpoint_dir`). Training resumes
automatically from the latest checkpoint found there if you restart with the same run dir.

### 2. Train the diffusion stage

The diffusion stage must be pointed at the regression checkpoint it conditions on. In the
training group config (e.g. `training/era_real_diffusion_patched.yaml`):

```yaml
io:
  regression_checkpoint_path: /path/to/regression/checkpoint/dir
```

then:

```bash
python src/hirad/training/train.py --config-name=training_era_real_diffusion_patched.yaml
```

### Overriding from the command line

Any field can be overridden without editing files, which is handy for sweeps and debug runs:

```bash
python src/hirad/training/train.py --config-name=training_era_real_regression.yaml \
    training.hp.lr=1e-4 training.hp.training_duration=50000 \
    dataset.start_date=2015-01-01 hydra.run.dir=./outputs/debug
```

### Available training configs

Analysis-input setups (`training_era_real_*`, `training_era_cosmo_*`), forecast-input setups
(`training_ifso1280_real_*`, with/without lead-time conditioning via the `_nolt` variants), and
the temporal `*_prevhr` variants. Prefer composing/selecting configs over editing code.

### Adapting training to your own problem

The config-only path — **new data on a regular grid, same model family** (any region, any size):

1. Build an input and a target **anemoi** store (see [Data](#data) and
   [custom datasets](#extending-hirad-gen-custom-datasets)).
2. Copy a `dataset/` config, point it at your stores, set `input_channel_names` /
   `output_channel_names` / `static_channel_names` to the variable names **in your stores**, the
   date ranges, and the `tp` transform constants.
3. Copy a top-level training config, swap in your `dataset/` group, pick a `model_size/`, and (for
   diffusion) set `regression_checkpoint_path`.

The model auto-sizes to the dataset's channels and grid, so a **different variable set, region, or
grid size** all work through config alone — as long as both stores are anemoi zarr with lat/lon on
a **regular (reshapeable) grid**. See
[Extending HiRAD-Gen → Case A](#extending-hirad-gen-custom-datasets) for the exact conditions.

> ⚠️ **An irregular / unstructured target grid does _not_ work through config alone.** The REAL
> target is stored on the unstructured ICON grid, so its ICON→rotated-lat-lon regrid and the paths
> to the remap indices/weights and lat/lon are **hardcoded** in `AnemoiDataset.__init__` (with
> `image_shape` hardcoded to `704×1088`), and the same regrid is applied in
> `TrainingManagerCorrDiff._preprocess_hr_target` (training) and inline in `inference/generate.py`
> (inference). A **new irregular grid** therefore needs code changes in all three; a new **regular**
> grid does not. Decoupling this regridding into a pluggable component is planned but not done. See
> [Extending HiRAD-Gen → Case B](#extending-hirad-gen-custom-datasets).

## Inference

Generate downscaled predictions with:

```bash
python src/hirad/inference/generate.py --config-name=<generate_config>.yaml
```

The top-level generate config composes a dataset (usually an `*_inference` variant with
`target_missing_as_zeros: true`), a sampler (`sampler/stochastic` or `sampler/deterministic`),
and a `generation/` group. The important `generation/` knobs:

```yaml
inference_mode: all          # "all" (regression + diffusion), "regression", or "diffusion"
num_ensembles: 16            # ensemble members per input time
seed_batch_size: 4           # batch size per gpu
randomize: True              # randomize seeds; if False, fixed seeds 0..num_ensembles-1 for each inference sample

# Patch-based sampling (same as model training parameters)
patching: True
patch_shape_x: 384
patch_shape_y: 384
overlap_pix: 4 # patch overlap in px-s
boundary_pix: 2 # boundary region for each patch

hr_mean_conditioning: True   # condition diffusion on the regression mean

# Which times to generate. Either an explicit list or a range + hourly step.
times: null
times_range: ['20211015-0000', '20211015-2300', 1]   # [start, end, hour_step], 'YYYYMMDD-HHMM'

io:
  res_ckpt_path: /path/to/diffusion/checkpoint/dir    # residual (diffusion) model
  reg_ckpt_path: /path/to/regression/checkpoint/dir   # mean (regression) model
  output_path: .
  output_format: 'torch'     # 'torch', 'grib', or 'both'
  grib_template_path: ''      # required only for GRIB output
```

Set the output location via Hydra:

```yaml
hydra:
  run:
    dir: /path/to/save/inference/output
```

Notes:

- Depending on `inference_mode`, you must supply the corresponding checkpoint(s). `all` needs
  both; `regression`/`diffusion` need only one.
- `output_format: 'torch'` writes raw tensors; `'grib'` writes GRIB (needs
  `grib_template_path`); `'both'` writes both.
- **Forecast precipitation.** For forecast inputs, `accumulate_tp` controls whether saved `tp`
  stays the per-step 1 h increment (default, matches the target for eval/snapshots) or is
  accumulated into a running total since forecast start (for GRIB/EvalML).
- **Temporal rollout.** With `prev_hr_conditioning` the generator can run an autoregressive
  rollout, feeding each step's output back as the previous-HR condition for the next
  (`rollout`, `history_source`). See the `generate_*_prevhr*` configs.

## Evaluation & visualization

Evaluation and visualization scripts depend on target grid size and description. In `eval_real.yaml` config, you can find neccessary information for evaluation scripts to work. When adapting to different region/grid, make sure to adapt the config and/or evaluation and visualization code. 

- **Snapshots.** `src/hirad/eval/snapshots.py` plots input / prediction / target maps for chosen
  times. Configured by `eval_*.yaml`, which point at the inference output directory:

  ```yaml
  inference_output_dir: '/path/to/generated/results'
  results_dir_name: 'name_of_plot_output_dir'
  times: null                 # or a list / times_range to plot a subset
  ```

- **Metrics.** Stand-alone scripts compute scores and diagnostic plots (bias by
  percentile, diurnal cycles, probability of exceedance, maps, …). These are tied to grid description and output structure. However, these can be adapted to work with other output structures and grids, as underlying metric logic is unchanged.

## Configuration layout

All behavior is composed from Hydra configs under `src/hirad/conf/`. Pick behavior by
choosing/composing configs rather than hardcoding.

- **Top-level entry configs** (passed as `--config-name`): `training_*`, `generate_*`. Each sets `hydra.run.dir` and a `defaults:` list composing the groups below.
- **Groups:**
  - `dataset/` — input/target stores, channels, transforms, dates (see [Data](#data)).
  - `model/` — network architecture and conditioning switches.
  - `model_size/` — width/depth presets.
  - `training/` — batch size, optimizer, scheduler, patching, model optimizations, I/O.
  - `sampler/` — `stochastic` vs `deterministic` diffusion sampling.
  - `generation/` — inference knobs (ensembles, patching, times, I/O).
  - `logging/` — metric logging (see below).

Naming conventions in the config names: `era`/`ifsn320`/`ifso1280`/`ifshres` = input source,
`real`/`cosmo` = target, `regression`/`diffusion` = stage, `patched` = patch-based,
`prevhr` = temporal conditioning, `nolt` = no lead-time conditioning.

### MLflow logging

Training can log metrics to MLflow. The logging group configs live in `src/hirad/conf/logging/`.
Set `method: mlflow` and, for a remote tracking server, a `uri`; otherwise the run is logged
locally in the output directory.

## Extending HiRAD-Gen: custom datasets

Everything in the pipeline is driven by a single dataset abstraction, so data is the main
extension point. There are really only **two cases**, and which one you are in is decided almost
entirely by one question: *is your target on a regular lat/lon grid stored as an anemoi dataset?*

If yes, the generic path handles it with no code (Case A). The only thing that needs code is an
**irregular / unstructured** target grid (like the REAL/ICON grid, whose regrid is hardcoded) or a
non-anemoi store (Case B).

### Case A: regular-grid anemoi stores — reuse the classes (config only)

If both your input and target stores are **anemoi zarr datasets on regular lat/lon grids**, you
need only a new `dataset/` config — no code, for any region or grid size. The model auto-sizes to
your channels and grid, and the whole input→target pipeline (interpolate, reshape, flip, normalize,
static conditioning) is grid-agnostic for regular grids. The conditions, concretely:

- **Anemoi stores, not plain zarr.** Both stores are opened with `anemoi.datasets.open_dataset`, so
  they must expose its API — `.statistics`, `.longitudes` / `.latitudes`, `.dates`, `.field_shape`.
  A raw zarr won't work.
- **Regular, reshapeable grids.** `field_shape` must be a 2-D `(H, W)` whose flat values reshape
  row-major (the loop does `reshape(field_shape).flip(-2)`), and the ensemble axis is squeezed
  (single member). Reduced-Gaussian / unstructured grids fail here — that is Case B.
- **Lat/lon present** on both stores (used to build the input→target interpolator and the geometry).
- **Aligned valid times.** Each input time must have a target at the same valid time, or set
  `input_frequency` to resample a finer input (see [Data](#data)).
- **Channel names match your stores.** Set `input_channel_names` / `output_channel_names` /
  `static_channel_names` to the variable names **as they exist in your zarr**; they are passed
  straight to `select=`. There is **no hardcoded channel naming** on the generic path — the
  era↔real precip-name map applies only to the REAL target. Box-Cox transforms, if any, reference
  channels by name through config.
- **Reuse the generic `type` token.** `type` is parsed as `<input>_<target>` and both tokens are
  whitelisted: the target must be `cosmo` or `real`, the input must be in the class's
  `VALID_INPUT_DATASETS` (`era5` for analysis, `ifs` for forecast). So label a regular-grid target
  `cosmo` — it is the generic non-real path, not tied to the actual COSMO region or size. For an
  analysis input this is the already-registered `anemoi_era5_cosmo` → `AnemoiDataset`; for a
  forecast input, add a one-line `known_datasets` entry `anemoi_ifs_cosmo → AnemoiForecastDataset`.

```python
known_datasets = {   # src/hirad/datasets/dataset.py
    "anemoi_era5_cosmo": ANEMOI_ERA5_COSMO,   # analysis input, regular-grid target
    "anemoi_era5_real":  ANEMOI_ERA5_REAL,     # analysis input, REAL/ICON target
    "anemoi_ifs_real":   ANEMOI_IFS_REAL,      # forecast input, REAL/ICON target
    ...
}
```

> ⚠️ **Known limitation — the cutout and interpolator are not spherical (no seam/pole wraparound).**
> To crop the input to the target, `AnemoiDataset.__init__` builds a bounding box from the target's
> **min/max** lon/lat ([anemoi_dataset.py:122-126](src/hirad/datasets/anemoi_dataset.py#L122-L126)),
> and that box plus the raw lon/lat feed the planar `GridData` interpolator. Longitudes are in the
> **[0, 360)** convention (a point "west of 0°" is stored as `360 − x`, *not* as a negative number),
> which is why `min_lon = max(0, …)` clamps the lower bound — it just keeps the box valid in that
> convention. The problem is that both the box and the interpolator treat lon/lat as a flat,
> monotonic axis, so they break for a region that **wraps around a coordinate seam**:
> - **Crossing the 0°/360° longitude seam** (e.g. a domain spanning 355°→5°): the points are
>   `…,358,359,0,1,…`, so `min(lon)≈0` and `max(lon)≈360` and the bounding box blows up to nearly the
>   whole globe. The interpolator, being planar, also treats 359° and 0° as ~359° apart instead of
>   adjacent.
> - **Crossing a pole**: latitude reverses at ±90°, so having a region cross any of the poles would lead to latitudes altering direction of change, making it impossible for linear interpolator to handle these cases.
>
> Switzerland and surroundings (a compact mid-latitude box,  not touching either the seam and the poles) avoids
> all of this, which is why it works. Doing it properly means performing the cutout **and** the
> interpolation in a spherical system that knows 0° and 360° are adjacent and handles pole crossings.
> This is future work and deliberately **not** changed here: the current behavior is baked into the
> already-trained CH model, so altering it would change that model's inputs. A new region that
> crosses the seam or a pole needs this reworked first.

### Case B: irregular grid or non-anemoi store - new/adapted class, plus the loop

You need code when the generic regular-grid path can't express your setup: an **irregular /
unstructured target grid** (needs a regrid, like REAL/ICON), a **non-anemoi store or storage
format**, or a **different input↔target pairing or normalization**. For an irregular target grid in
particular, the regrid is duplicated across three places that must change together (this is exactly
the REAL/ICON coupling — treat its code as the template):

1. **The dataset class** — `AnemoiDataset.__init__` loads the REAL grid lat/lon and the
   ICON→rotated-lat-lon remap indices/weights from `/capstor/.../real_grid_info/` and hardcodes
   `image_shape()` to `704×1088`. Add your grid's geometry and remap operator here.
2. **The training manager** - `TrainingManagerCorrDiff._preprocess_hr_target` applies the regrid,
   `trim_edge` crop and flip for the REAL branch; adapt (or add) that branch for your grid.
3. **The inference loop** — `inference/generate.py` repeats the *same* regrid/trim/flip inline (not
   shared with the training manager), for both the target and the previous-HR history field, and
   moves the index/weight tensors to device at startup.

Change one without the others and training↔inference consistency breaks silently. (Decoupling this
regridding into a single pluggable component is planned but not done.) Register the new class's
`type` in `known_datasets` as usual.

**To write the class itself, two routes:**

- **Subclass `AnemoiDataset` / `AnemoiForecastDataset`** when you are still backed by anemoi
  stores but need to change *how* input and target are paired or normalized. `AnemoiForecastDataset`
  is itself a worked example — it overrides only the handful of methods that differ between the
  analysis and forecast layouts:

  | Method | Purpose |
  |--------|---------|
  | `_open_input_dataset` | how the input store is opened/subset (forecast can't use `select=`) |
  | `_align_input_output` | how input samples are paired with target valid times |
  | `_get_target_by_position` | fetch the target for a sample position |
  | `_fallback_input_statistics` | reorder/compute input stats to channel order |
  | `previous_index` | the temporal predecessor of a sample (for prev-HR conditioning) |
  | `__getitem__`, `__len__`, `time` | per-sample access and the sample timeline |

- **Subclass `DownscalingDataset` directly** (in
  [`src/hirad/datasets/base.py`](src/hirad/datasets/base.py)) for a non-anemoi source.
  `era5_real.py` / `era5_cosmo.py` are examples. You must implement the abstract interface:

  ```python
  class DownscalingDataset(torch.utils.data.Dataset, ABC):
      def longitude(self) -> np.ndarray: ...        # target-grid lon
      def latitude(self)  -> np.ndarray: ...        # target-grid lat
      def input_channels(self)  -> List[ChannelMetadata]: ...   # drives model in-channels
      def output_channels(self) -> List[ChannelMetadata]: ...   # drives model out-channels
      def time(self) -> List: ...                   # sample timeline, 'YYYYMMDD-HHMM'
      def image_shape(self) -> Tuple[int, int]: ... # target (H, W)
  ```

  and optionally override the hooks with sensible defaults: `normalize_input` / `denormalize_input`
  / `normalize_output` / `denormalize_output` (physical ↔ normalized), `previous_index`
  (temporal conditioning; return `None` if unsupported), and `info`.

  **`__getitem__` contract.** Return a tuple `(target, input, date_str)` of tensors in **physical
  units**, each on its own native grid and in the native orientation of its store. **You do not
  interpolate, regrid, flip, or normalize in the dataset** — the loop does all of that (see
  [the preprocessing pipeline](#the-preprocessing-pipeline-what-the-loop-does-to-your-data)
  below). Append optional trailing items, in this order, when the corresponding switch is on: an
  integer lead-time label (`provides_lead_time`), then `(prev_target, has_prev)`
  (`provides_prev_target`). Finally add the new `type` string to `known_datasets`.

#### The preprocessing pipeline (what the loop does to your data)

**This is important to have in mind.** `__getitem__` returns *raw* fields; a substantial pipeline
then transforms them before they reach the network, and it is driven entirely by methods and
attributes your dataset exposes. If you write a custom dataset you must return data in exactly the
form this pipeline expects, so understand every step. It lives in
[`TrainingManagerCorrDiff.load_and_preprocess_batch`](src/hirad/training/training_manager.py) (and
the matching inference path), with the final conditioning assembled in
[`losses/loss.py`](src/hirad/losses/loss.py). Per batch, in order:

1. **Unpack optional trailing fields.** `img_clean, img_lr, *rest = next(iterator)`. The loop pops
   the optional items off the **end** of `rest`, in reverse of how `__getitem__` appended them:
   `(prev_target, has_prev)` first (if `provides_prev_target`), then the `lead_time_label` (if
   `provides_lead_time`). Whatever remains is the date info passed to `make_time_grids`. Order
   matters — get it wrong and the wrong tensor is read as the date.

2. **Input → target grid (interpolation).** The low-res input is on a *different, generally
   coarser* grid than the target. It is mapped onto the **target grid** by
   `self.dataset.interpolator` — a [`GridData`](src/hirad/utils/dataset_utils.py) object built in
   `AnemoiDataset.__init__` from the input lon/lat and the target lon/lat. `GridData` does
   Delaunay-triangulation **barycentric (piecewise-linear) interpolation** (equivalent to
   `scipy.interpolate.griddata(method="linear")`, precomputed once for the fixed point sets). So
   the network never sees the native input grid — it sees the input resampled onto the HR grid.
   The `INPUT_MARGIN_DEGREES = 0.5` crop in `AnemoiDataset` exists to avoid NaNs at the
   interpolation boundary. Both the input-area bounding box and `GridData` are **planar, not
   spherical** (no 0°/360° seam or pole wraparound) — see the longitude-handling limitation in
   [Extending HiRAD-Gen → Case A](#extending-hirad-gen-custom-datasets) before porting to a region
   that crosses the seam or a pole.

3. **Reshape + orientation.** The interpolated input is reshaped to `image_shape()` `(H, W)` and
   **flipped vertically** (`.flip(-2)`): the stores are bottom-left origin, the model works
   top-left origin. Your `image_shape()` and the native row order therefore both matter.

4. **Normalize input.** `self.dataset.normalize_input(...)` applies any per-channel transform
   (e.g. Box-Cox on `tp`) then standardizes with the input mean/std. Stats come from the input
   store, or an override store for cross-source inference (see [Data](#data)).

5. **High-res target.** `_preprocess_hr_target` handles the target (and the previous-step history
   field identically):
   - **REAL target:** `regrid_icon_to_rotlatlon(img, regrid_indices_real, regrid_weights_real)` —
     the target is stored on the **ICON native grid** and is remapped to the rotated-lat-lon model
     grid with precomputed index/weight arrays (loaded from the grid-info files in the dataset),
     then `trim_edge`-cropped and flipped. This is done **on the GPU in the loop**, not in
     `__getitem__`, for throughput.
   - **COSMO target:** reshaped to `image_shape()` and flipped.
   - Then `normalize_output`.

6. **Cyclic time features.** If `n_month_hour_channels > 0`, `make_time_grids(date_str, ...)`
   builds multi-frequency sin/cos encodings of hour-of-day and month-of-year from the date string;
   these become extra conditioning channels.

7. **Static channels.** `get_static_data()` returns the normalized static field(s) (e.g. orography
   `FIS` / `hsurf`), prepared once in `__init__` to `(C, H, W)` on the target grid — regridded for
   an irregular (REAL/ICON) target, reshaped for a regular one — and flipped to model orientation
   here.

8. **Device / dtype / memory format.** Everything is moved to the GPU in `input_dtype`, in
   `channels_last` when apex GroupNorm is used.

Only after all of this does the **loss** assemble the conditioning tensor `y_lr` the U-Net sees,
by channel-concatenating, in this order:

```
[ regression mean (y_mean, diffusion + hr_mean_conditioning only),
  interpolated low-res input,
  static channels,
  cyclic date embedding,
  previous-HR history (prev_hr, temporal conditioning only) ]
```

and the network then appends its own **positional grid embedding** (`N_grid_channels`) and, with
`lead_time_mode`, a broadcast **lead-time embedding**. This is why `create_model` computes
`img_in_channels` as the sum of input + static + time + (optional mean/prev-HR) + grid (+ lead)
channels — all derived from your dataset, never hardcoded.

> **Patching caveat.** With patch-based training the conditioning is split into two paths: the
> **cropped** path (`input=y_lr`: regression mean, low-res, static, history) is cropped to the
> patch and stays pixel-aligned with the residual target, while the **global-context** path
> (`additional_input=img_lr`: low-res + static + date) is bilinearly interpolated
> to the patch size.

#### Checklist: the surface a dataset must expose

Everything the pipeline above needs comes from the dataset. A custom dataset must provide:

- **Geometry/metadata:** `longitude()`, `latitude()`, `image_shape()`, `input_channels()`,
  `output_channels()`, `static_channels()`, `time()` — these size the model (channel counts +
  grid) and drive interpolation; nothing is hardcoded in the model config.
- **The interpolator:** a `self.interpolator` (`GridData`) mapping input lon/lat → target lon/lat
  (built in `AnemoiDataset.__init__`; replicate it if you subclass `DownscalingDataset` directly).
- **Normalization:** `normalize_input/output`, `denormalize_input/output`, `normalization_stats()`,
  and the per-channel mean/std used by them.
- **Static data:** `get_static_data()` returning the (normalized, target-grid) static field or
  `None`.
- **Optional conditioning flags:** set `provides_lead_time` / `provides_prev_target` and implement
  `previous_index()` — the loop detects these and wires the extra inputs, so a conditioning feature
  turns on with just the dataset flag plus the matching `model/` switch, no loop changes.
- **For the REAL target path:** `regrid_indices_real` / `regrid_weights_real` / `trim_edge` (the
  loop uses them in `_preprocess_hr_target`).

### Beyond data: losses and models

Loss functions live in [`src/hirad/losses/`](src/hirad/losses/) and models in
[`src/hirad/models/`](src/hirad/models/) (`song_unet.py`, `unet.py`, `preconditioning.py`). A new
loss or preconditioner is wired in the same config-driven way (referenced by name from the
`model/` and `training/` groups) and the training manager
([`src/hirad/training/training_manager.py`](src/hirad/training/training_manager.py)) is the place
that assembles dataset → model → loss for a run.

## Running on CSCS Alps (clariden/santis)

> This section is specific to the CSCS **Alps** cluster (clariden/santis). The model, data and
> config interfaces above do not depend on it.

**Everything runs through SLURM.** Do not run on the login node — Jobs run inside a prebuilt container; no extra installation is needed.

- **Container.** The environment is defined by `ci/edf/modulus_env.toml` and passed to `srun`
  via `--environment=./ci/edf/modulus_env.toml`. The squashed image lives on Alps at
  `/capstor/scratch/cscs/pstamenk/corr_diff.sqsh`.
- **Entrypoints inside `srun`.** Scripts either `pip install -e .` or set
  `PYTHONPATH=$PWD/src` before calling the Python entrypoint. (For SLURM-array jobs where tasks
  would race on a shared install, prefer `PYTHONPATH`.)
- **Accounts/partitions vary** — read the existing `.sh` before copying one. Production scripts
  use `--partition=normal`; debug/test scripts use `--partition=debug` with a short `--time`.
  Point `--output`/`--error` at a **shared** path (e.g. `./logs/`), never `/tmp` (node-local and
  unreadable from the login node).
- **Multi-node** uses `DISTRIBUTED_INITIALIZATION_METHOD=SLURM` with `MASTER_ADDR`/`MASTER_PORT`
  derived from `scontrol` — already wired in the runner scripts.

Ready-made runners (start from these rather than reinventing):

| Task | Script | Entrypoint |
|------|--------|------------|
| Regression training | `src/hirad/train_regression.sh` | `training/train.py` |
| Diffusion training | `src/hirad/train_diffusion.sh` | `training/train.py` |
| Generation / inference | `src/hirad/generate.sh` | `inference/generate.py` |
| Snapshots | `src/hirad/snapshots.sh` | `eval/snapshots.py` |
| Evaluation | `eval_precip.sh`, `eval_temp.sh`, `eval_wind.sh` | `diurnal_cycle_wind.py`, … |

Each runner has a header block to edit before submitting:

```bash
### OUTPUT ###
#SBATCH --output=./logs/<job>.log
#SBATCH --error=./logs/<job>.err

### ENVIRONMENT ###
#SBATCH -A <your_account>

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/<entrypoint>.py --config-name=<your_config>.yaml
"
```

Submit with `sbatch src/hirad/<script>.sh`. For monthly batch generation there is a helper,
`src/hirad/submit_monthly.sh START_MONTH END_MONTH`, which drives `generate.sh` as a SLURM array.

## MeteoSwiss-internal tools

> This section requires MeteoSwiss-internal infrastructure (internal zarr stores, grid-info
> files, the Balfrin cluster, and EvalML). External users without access can skip it; the
> training/inference flow above works with any anemoi input/target stores and any output grid.

### Internal data and grid resources

The example configs reference internal stores and resources, e.g.:

- Target: `/capstor/store/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr`
- Input (ERA5/N320 analysis): `/capstor/store/mch/msopr/ml/datasets/aifs-ea-an-oper-...-with-era51.zarr`
- RealCH1 grid info (lat/lon + ICON→rot-lat-lon remap indices/weights):
  `/capstor/store/cscs/pasc/c38/real_grid_info/`
- Model checkpoints: `/capstor/store/cscs/pasc/c38/model_checkpoints/`
- GRIB templates for inference output: `/capstor/store/cscs/pasc/c38/evalml/resources/inference/templates`

Replace these with your own anemoi stores to run elsewhere. The grid-info files above are needed
**only for the REAL/ICON (irregular) target**, whose regrid is hardcoded in the dataset class, the
training manager and `generate.py`; a new **irregular** grid needs code changes across all three. A
new **regular**-grid anemoi target needs none of this — it runs config-only (see
[Extending HiRAD-Gen → Case A](#extending-hirad-gen-custom-datasets)).

### EvalML

EvalML is MeteoSwiss's evaluation framework. Displaying HiRAD-Gen results in EvalML is
work-in-progress and runs on the **Balfrin** cluster.

After running inference with GRIB output, copy the `grib` directory to Balfrin. (Its
subdirectories are dates, each containing a further `grib` subdirectory.)

```bash
# On your local machine
scp -r santis:/capstor/scratch/cscs/<user>/outputs/generation/<run>/grib/ .
scp -r grib/ balfrin:/store_new/mch/msopr/ml/evalml/hirad_grib_data/
```

On Balfrin:

1. Clone https://github.com/MeteoSwiss/evalml/ and check out `feat/hirad-integration` (adds
   running models directly from GRIB).
2. Follow EvalML's [installation](https://github.com/MeteoSwiss/evalml/tree/feat/hirad-integration#installation).
3. Edit `tests/integration/configs/hirad.yaml` — set the base `dates` to the init times in your
   data (and optionally `steps` under `grib_forecaster` and `config_label`).
4. Run the experiment:
   ```bash
   evalml experiment tests/integration/configs/hirad.yaml
   ```
   Results appear under `output/results/<date>_hirad-forecast-test-1.0-<key>`.
</content>
</invoke>
ntent>
</invoke>

Your results will appear in relative path `output/results/<date>_hirad-forecast-test-1.0-<some key thing>`

## EvalML

Running HiRAD in EvalML involves two parts: generating the GRIB output (either
from inference `generate.py` directly, or converting from existing torch output),
and then running the evaluation in EvalML.

### Generating GRIB output

The directory structure
is a bit weird: it'll be the `<root>/<pred_type>/YYYYMMDDHHMM/grib/YYYYMMDDHHMM_HH.grib`
This is just what evalML expects.

#### New inference run

If you are performing a new inference run, you can in the `/conf/generation/*.yaml` file,
set `io: output_format: grib` or `io: output_format: both` (produces both torch and grib).
Then run inference as usual.
This will put all GRIB output into a directory called `grib` under the `io: output_path`.


#### Existing inference run
If you have already performed inference and only have `torch` output, you can
run a script to convert it to GRIB format. See the `src/hirad/torch_to_grib.sh` 
script to run. Update the SRC and DST directories and indicate what base times
you want to use. (script generates a month worth of basetimes, or you can say "all")


#### Copying to `balfrin`/`linard`
EvalML currently only runs on balfrin/linard, so you will need to copy it over
to make it usable.

### Running EvalML

#### Installing

The implementation to enable EvalML to accept externally-generated GRIB files
is not yet merged to main; it lives in a branch `feat/hirad-integration`. 

From your $SCRATCH directory, check out the repo:

`git clone -b feat/hirad-integration https://github.com/MeteoSwiss/evalml.git`

Then use the installation instructions in the EvalML README:
Install uv
`curl -LsSf https://astral.sh/uv/install.sh | sh`

Install project
`uv sync`

Activate
`source .venv/bin/activate`

#### Configuring

The HiRAD configuration is in `tests/integration/configs/hirad.yaml`.
Each `-grib_forecaster` block will add another experiment for comparison.
Change `root` to be the location where you copied the grib for each pred type.

You can also change start and end dates (I suggest running a test with a couple days
of base times just to make sure it works)

It's also a good idea to change the `config_label` to differentiate between runs--
this will be part of the output folder name. But if you forget to change the name,
it'll still create a hash and use the run date to name the folder so there won't
be any overwritten issue.

#### Running

To run, call:
`evalml experiment tests/integration/configs/hirad.yaml`

It will set up a Snakemake graph to execute dependencies and run.

If the workflow fails, it is a bit annoying to debug manually; I recommend
copying the snakemake log file path and asking Claude to find the failed job
and error log.

Outputs (plots, scorecards) will show up in the `outputs/results` subdirectory.
A new directory is created for each run.
