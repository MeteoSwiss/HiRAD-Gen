# HiRAD-Gen

HiRAD-Gen is short for high-resolution atmospheric downscaling using generative models. This repository contains the code and configuration required to train and use the model.

[Inference - clariden/santis](#running-inference-on-alps)  
[Setup - clariden/santis](#setup-claridensantis)   
[Regression training - clariden/santis](#run-regression-model-training-alps)  
[Diffusion training - clariden/santis](#run-diffusion-model-training-alps)  

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
