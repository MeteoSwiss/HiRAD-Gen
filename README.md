# HiRAD-Gen

HiRAD-Gen is short for high-resolution atmospheric downscaling using generative models. This repository contains the code and configuration required to train and use the model.

[Showcase](#Showcase)  
[Setup - clariden/santis](#setup-claridensantis)   
[Inference - clariden/santis](#running-inference-on-alps)  
[Regression training - clariden/santis](#run-regression-model-training-alps)  
[Diffusion training - clariden/santis](#run-diffusion-model-training-alps)  

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

### Ensemble Total Preceipitatin 1h

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

## Setup clariden/santis container environment
Container environment setup needed to run training and inference experiments on clariden/santis is contained in this repository under `ci/edf/modulus_env.toml`. Image squash is on clariden/alps under `/capstor/scratch/cscs/pstamenk/corr_diff.sqsh`. All the jobs can be run using this environment without additional installations and setup.

## Inference

### Running inference on Alps

1. Script for running the inference is in `src/hirad/generate.sh`. 
Inside this script set the following:
```bash
### OUTPUT ###
#SBATCH --output=path_to_output_log
#SBATCH --error=path_to_output_error
```
```bash
#SBATCH -A compute_group
```
```bash
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/inference/generate.py --config-name=main-config-file-in-src/hirad/conf.yaml
"
```

2. Set up the following config files in `src/hirad/conf`:

- In main config file (by default `generate_era_real.yaml`) set:
```
hydra:
  run:
    dir: your_path_to_save_inference_output
```
- In generation config file (by default `generation/era_real.yaml`):
Choose the inference mode:
```
inference_mode: all/regression/diffusion
```
by default `all` does both regression and diffusion. Depending on mode, regression and/or diffusion model pretrained weights should be provided:
```
io:
  res_ckpt_path: path_to_directory_containing_diffusion_training_model_checkpoints
  reg_ckpt_path: path_to_directory_containing_regression_training_model_checkpoints
```
Finally, from the dataset, subset of time steps can be chosen to do inference for.

One way is to list steps under `times:` in format `%Y%m%d-%H%M` for era5_cosmo dataset.

The other way is to specify `times_range:` with three items: first time step (`%Y%m%d-%H%M`), last time step (`%Y%m%d-%H%M`), hour shift (int). Hour shift specifies distance in hours between closest time steps for specific dataset.

3. Submit the job with:
```bash
sbatch src/hirad/generate.sh
```

### Visualizing results

After generation is finished, visualization of results can be done using `src/hirad/snapshots.sh`. Set:
```bash
### OUTPUT ###
#SBATCH --output=path_to_output_log
```
```bash
### ENVIRONMENT ####
#SBATCH -A compute_group
```
```bash
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/eval/snapshots.py --config-name=src/hirad/conf/config-file-in-src/hirad/conf.yaml
"
```
In config file (by default `eval_real.yaml`) set:
```bash
# Path to the inference output directory
inference_output_dir: '/path/to/generated/results/directory'
results_dir_name: 'name_of_directory_to_save_output_plots'
```
If you want to generate plots for subset of times from inference set (follow same convection as in generate config):
```
times: list of times to visualize
times_range: [start time, end time, time step] to visualize
```

Other setting can be changed according to output grid.

Submit the job with:
```bash
sbatch src/hirad/snapshots.sh
```

### Evaluation of generated data

Evaluation of generated samples can be done using `src/hirad/eval_precip.sh` and `src/hirad/eval_wind.sh`. Set:
```bash
### OUTPUT ###
#SBATCH --output=path_to_output_log

### ENVIRONMENT ####
#SBATCH -A compute_group

### CONFIG ###
CONFIG_NAME="src/hirad/conf/config_file.yaml"
```
Default config file is the same as for visualization `eval_real.yaml`, and requires to set the same fileds. In both `eval_precip.sh` and `eval_wind.sh` there are several python scripts called. They are all commented out by default. Comment out the ones you want to run.

Submit jobs with:
```bash
sbatch src/hirad/eval_precip.sh
sbatch src/hirad/eval_wind.sh
```

## Training

### Run regression model training (Alps)

1. Script for running the training of regression model is in `src/hirad/train_regression.sh`. Here, you can change the sbatch settings.
Inside this script set the following:
```bash
### OUTPUT ###
#SBATCH --output=path_to_output_log
#SBATCH --error=path_to_output_error
```
```bash
#SBATCH -A compute_group
```
```bash
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/training/train.py --config-name=main-config-file-in-src/hirad/conf.yaml
"
```

2. Set up the following config files in `src/hirad/conf`:

- In main config file (by default `training_era_real_regression.yaml`) set:
```
hydra:
  run:
    dir: your_path_to_save_training_outputs
```
- All other parameters for training regression can be changed in the main config file and config files the main config is referencing (default values are working for debugging purposes).

3. Submit the job with:
```bash
sbatch src/hirad/train_regression.sh
```

### Run diffusion model training (Alps)
Before training diffusion model, checkpoint for regression model has to exist.

1. Script for running the training of diffusion model is in `src/hirad/train_diffusion.sh`. Here, you can change the sbatch settings. Inside this script set the following:
```bash
### OUTPUT ###
#SBATCH --output=your_path_to_output_log
#SBATCH --error=your_path_to_output_error
```
```bash
#SBATCH -A your_compute_group
```
```bash
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/training/train.py --config-name=main-config-file-in-src/hirad/conf.yaml
"
```

2. Set up the following config files in `src/hirad/conf`:

- In main config file (by default `training_era_real_diffusion_patched.yaml`) set:
```
hydra:
  run:
    dir: your_path_to_save_training_output
```
- In training config file (by default `training/era_real_diffusion_patched.yaml`) set:
```
io:
    regression_checkpoint_path: path_to_directory_containing_regression_training_model_checkpoints
```
- All other parameters for training regression can be changed in the main config file and config files the main config is referencing (default values are working for debugging purposes).

3. Submit the job with:
```bash
sbatch src/hirad/train_diffusion.sh
```

## MLflow logging

During training MLflow can be used to log metrics.
Logging config files for regression and diffusion are located in `src/hirad/conf/logging/`. Set `method` to `mlflow` and specify `uri` if you want to log on remote server, otherwise run will be logged locally in output directory. Other options can also be modified here.

## EvalML

Displaying results in EvalML is work-in-progress.

After running inference and generating GRIB, copy files in the `grib` directory over to Balfrin.
(The subdirectories of the `grib` directory should be dates, which then each a further `grib` subdirectory inside of it. Don't ask.)

On your local machine:
```
scp -r santis:/capstor/scratch/cscs/mmcgloho/outputs/generation/ifsn320_real_results_tp_deaccum/grib/ .
scp -r grib/ balfrin:/store_new/mch/msopr/ml/evalml/hirad_grib_data/
```

On balfrin:

Download the git repo from https://github.com/MeteoSwiss/evalml/

Check out branch `feat/hirad-integration` which has the implementation for using models directly from GRIB.

Follow [installation](https://github.com/MeteoSwiss/evalml/tree/feat/hirad-integration#installation) instructions for EvalML.

Edit the file `tests/integration/configs/hirad.yaml` to adjust the base `dates` field to use init times in the data you want to evaluate. (You can also adjust the `steps` under the `grib_forecaster`). You might want to edit the config_label as well. 

Run the experiment:
```
evalml experiment tests/integration/configs/hirad.yaml 
```

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
run a script to convert it to GRIB format.

```

```
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
