# Running anemoi downscaling for HiRAD

## Installing 

Check out the `hirad-60` branch.

You can probably use Mary's venv:

```
source /users/mmcgloho/hirad-gen/HiRAD-Gen/.venv/bin/activate
```


## Example use case

The example use case is from Yoël Zérah from the anemoi downscaling group. It is a task of downscaling ERA5 to CERRA in the European region. 

### Config

The config lives in `src/anemoi-core-ds-collab/training/src/anemoi/training/config/sample_training_config_santis.yamlsrc/anemoi-core/ds-collab/training/sample_training_config_santis.yaml`

### Data dependencies

Input data (including residuals inputs) are in `/capstor/scratch/cscs/mmcgloho/anemoi-downscaling/downscaling-data`

Other data storage (checkpoints, etc) is also under the `anemoi-downscaling` directory.


### Run

To run:

```
sbatch src/anemoi-core-ds-collab/training/train_sample_santis.sh
```

### Experiments

#### Running an overfitting experiment
To train on a single data point, change line in `src/anemoi-core-ds-collab/training/src/anemoi/training/data/dataset/downscalingdataset.py` to iterate over index 0 only. (This may already be enabled)

Joffrey had instructed me to do this. (It should also be possible to do this by changing the start/end in the dataloader: training: config, but for some reason I get an error doing this.)

#### Switching between deterministic and probabilistic mode

"Deterministic" training means running the diffusion model on a constant noise. This will help the model converge faster, but it will learn the noise (fun fact: this is what CorrDiff code originally did, and why our first models performed reasonably well but had some weird artifacts). Therefore, even in an overfitting situation the error will never reach 0.

Probabilistic means changing the noise in each training step. As of 12. May, we have not yet been able to get a model to converge in probabilistic mode.

To change between deterministic/probabilistic, the config is under top-level `training:`. Set `deterministic: True` and `training_approach: deterministic` for determinstic, or `deterministic: False` and `training_approach: probabilistic_low_noise` for probabilistic.


#### Running from checkpoint

To run from a checkpoint, set `training: run_id: ` to be the hash value of the run you want to use the checkpoint from.




## ERA-COSMO use case

TODO: Document
