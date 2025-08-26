# HiRAD-Gen

HiRAD-Gen is short for high-resolution atmospheric downscaling using generative models. This repository contains the code and configuration required to train and use the model.

[Setup clariden/santis](#setup-claridensantis)   
[Regression training - clariden/santis](#run-regression-model-training-alps)  
[Diffusion training - clariden/santis](#run-diffusion-model-training-alps)  
[Inference - clariden/santis](#running-inference-on-alps)  
[Installation - uenv/venv - deprecated](#installation-alps-uenvvenv---deprecated)

## Setup clariden/santis container environment
Container environment setup needed to run training and inference experiments on clariden/santis is contained in this repository under `ci/edf/modulus_env.toml`. Image squash is on clariden/alps under `/capstor/scratch/cscs/pstamenk/hirad.sqsh`. All the jobs can be run using this environment without additional installations and setup.  

## Training

### Run regression model training (Alps)

1. Script for running the training of regression model is in `src/hirad/train_regression.sh`. Here, you can change the sbatch settings.
Inside this script set the following:
```bash
### OUTPUT ###
#SBATCH --output=your_path_to_output_log
#SBATCH --error=your_path_to_output_error
```
```bash
#SBATCH -A your_compute_group
```
```bash
srun bash -c "
    . ./{your_env_name}/bin/activate
    python src/hirad/training/train.py --config-name=training_era_cosmo_regression.yaml
"
```

2. Set up the following config files in `src/hirad/conf`:

- In `training_era_cosmo_regression.yaml` set:
```
hydra:
  run:
    dir: your_path_to_save_training_outputs
```
- All other parameters for training regression can be changed in the main config file `training_era_cosmo_regression.yaml` and config files the main config is referencing (default values are working for debugging purposes).

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

2. Set up the following config files in `src/hirad/conf`:

- In `training_era_cosmo_diffusion.yaml` set:
```
hydra:
  run:
    dir: your_path_to_save_training_output
```
- In `training/era_cosmo_diffusion.yaml` set:
```
io:
    regression_checkpoint_path: path_to_directory_containing_regression_training_model_checkpoints
```
- All other parameters for training regression can be changed in the main config file `training_era_cosmo_diffusion.yaml` and config files the main config is referencing (default values are working for debugging purposes).

3. Submit the job with:
```bash
sbatch src/hirad/train_diffusion.sh
```

## Inference

### Running inference on Alps

1. Script for running the inference is in `src/hirad/generate.sh`. 
Inside this script set the following:
```bash
### OUTPUT ###
#SBATCH --output=your_path_to_output_log
#SBATCH --error=your_path_to_output_error
```
```bash
#SBATCH -A your_compute_group
```
```bash
srun bash -c "
    . ./{your_env_name}/bin/activate
    python src/hirad/inference/generate.py --config-name=generate_era_cosmo.yaml
"
```

2. Set up the following config files in `src/hirad/conf`:

- In `generate_era_cosmo.yaml` set:
```
hydra:
  run:
    dir: your_path_to_save_inference_output
```
- In `generation/era_cosmo.yaml`:
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

- In `dataset/era_cosmo_inference.yaml` set the `dataset_path` if different from default. Make sure that specified times or times_range is contained in dataset_path.

3. Submit the job with:
```bash
sbatch src/hirad/generate.sh
```

## MLflow logging

During training MLflow can be used to log metrics.
Logging config files for regression and diffusion are located in `src/hirad/conf/logging/`. Set `method` to `mlflow` and specify `uri` if you want to log on remote server, otherwise run will be logged locally in output directory. Other options can also be modified here.

## Installation (Alps uenv/venv) - deprecated

To set up the environment for **HiRAD-Gen** on Alps supercomputer, follow these steps:

1. **Start the PyTorch user environment**:
   ```bash
   uenv start pytorch/v2.6.0:v1 --view=default
   ```

2. **Create a Python virtual environment** (replace `{env_name}` with your desired environment name):
   ```bash
   python -m venv ./{env_name}
   ```

3. **Activate the virtual environment**:
   ```bash
   source ./{env_name}/bin/activate
   ```

4. **Install project dependencies**:
   ```bash
   pip install -e .
   ```

This will set up the necessary environment to run HiRAD-Gen within the Alps infrastructure.
