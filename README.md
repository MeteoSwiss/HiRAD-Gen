# HiRAD-Gen

HiRAD-Gen is short for high-resolution atmospheric downscaling using generative models. This repository contains the code and configuration required to train and use the model.

## Installation (Alps)

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

## Run regression model training (Alps)

1. Script for running the training of regression model is in `src/hirad/train_regression.sh`. 
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

2. Setup the following config files in `src/hirad/conf`:

- In `training_era_cosmo_regression.yaml` set:
```
hydra:
  run:
    dir: your_path_to_save_training_output
```
- In `training/era_cosmo_regression.yaml` set:
```
hp:
    training_duration: number of samples to train for (set to 4 for debugging, 512 fits into 30 minutes on 1 gpu with total_batch_size: 4)
```
- In `dataset/era_cosmo.yaml` set the `dataset_path` if different from default.

3. Submit the job with:
```bash
sbatch src/hirad/train_regression.sh
```

## Run diffusion model training (Alps)
Before training diffusion model, checkpoint for regression model has to exist.

1. Script for running the training of diffusion model is in `src/hirad/train_diffusion.sh`. 
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
    python src/hirad/training/train.py --config-name=training_era_cosmo_diffusion.yaml
"
```

2. Setup the following config files in `src/hirad/conf`:

- In `training_era_cosmo_diffusion.yaml` set:
```
hydra:
  run:
    dir: your_path_to_save_training_output
```
- In `training/era_cosmo_regression.yaml` set:
```
hp:
    training_duration: number of samples to train for (set to 4 for debugging, 512 fits into 30 minutes on 1 gpu with total_batch_size: 4)
io:
    regression_checkpoint_path: path_to_directory_containing_regression_training_model_checkpoints
```
- In `dataset/era_cosmo.yaml` set the `dataset_path` if different from default.

3. Submit the job with:
```bash
sbatch src/hirad/train_diffusion.sh
```