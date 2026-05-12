# Manual Inference

This package produces `predictions.nc` from a checkpoint using either:
- the Anemoi dataloader
- a prebuilt input bundle (`.nc`) from MARS/GRIB

Strict new-stack policy:
- `y` truth must be present in `predictions.nc`.
- `from-bundle` is the production path.
- `from-dataloader` is debug-only and requires `--debug-from-dataloader`.
- bundle build requires target truth by default. `--allow-missing-target-unsafe` is an explicit
  prediction-only escape hatch and produces non-canonical bundles without `target_hres_*`.

## Notebooks (Super Simple)
- `manual_inference/notebooks/00_manual_inference_overview.ipynb`
- `manual_inference/notebooks/01_prediction_from_dataloader.ipynb`
- `manual_inference/notebooks/02_prediction_from_bundle.ipynb`
- `manual_inference/notebooks/03_build_bundle.ipynb`

## Entry Points
- `manual_inference/prediction/predict.py` (CLI module)
- `manual_inference/input_data_construction/bundle.py` (GRIB → bundle)

## Common Commands
From dataloader:
```bash
python -m manual_inference.prediction.predict from-dataloader \
  --debug-from-dataloader \
  --name-ckpt <exp_or_ckpt> \
  --idx 0 --n-samples 1 --members 0 \
  --extra-args-json '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'
```

From input bundle:
```bash
python -m manual_inference.prediction.predict from-bundle \
  --name-ckpt <exp_or_ckpt> \
  --bundle-nc /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc \
  --extra-args-json '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'
```

Build bundle:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --target-sfc-grib /path/hres_target_sfc.grib \
  --target-pl-grib  /path/hres_target_pl.grib \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

Unsafe bundle build when target GRIBs are missing:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --allow-missing-target-unsafe \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

## Output
If `--out` is not set, predictions are written to:
`/home/ecm5702/hpcperm/experiments/<name_exp>/predictions.nc`
