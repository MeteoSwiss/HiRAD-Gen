**Prediction Workflow**
This folder produces `predictions.nc` from a checkpoint using either:
1. the Anemoi dataloader, or
2. a prebuilt **input bundle** (`.nc`) made from MARS/GRIB.

**Simple notebooks**
- `manual_inference/notebooks/01_prediction_from_dataloader.ipynb`
- `manual_inference/notebooks/02_prediction_from_bundle.ipynb`
- `manual_inference/notebooks/03_build_bundle.ipynb`

**Default Output Location**
If `--out` is not set, predictions are written to:
`/home/ecm5702/hpcperm/experiments/<name_exp>/predictions.nc`

**Strict new-stack policy**
- `y` truth is mandatory in output `predictions.nc`.
- `from-bundle` is the production path.
- `from-dataloader` is debug-only and requires `--debug-from-dataloader`.
- bundle build requires target truth by default.
- `build-bundle`, `from-bundle`, and `generate_predictions_25_files.py` may use
  `--allow-missing-target-unsafe` only for explicit prediction-only recovery output.
- If you explicitly need prediction-only recovery output from an incomplete bundle, `from-bundle`
  can use `--allow-missing-target-unsafe` to write `y` as all-NaN. Treat that output as
  non-canonical for truth-aware evaluation.

**Commands**

1. From dataloader:
```bash
python -m manual_inference.prediction.predict from-dataloader \
  --debug-from-dataloader \
  --name-ckpt <exp_or_ckpt> \
  --idx 0 --n-samples 1 --members 0 \
  --extra-args-json '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'
```

2. From input bundle (preferred for MARS data):
```bash
python -m manual_inference.prediction.predict from-bundle \
  --name-ckpt <exp_or_ckpt> \
  --bundle-nc /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc \
  --extra-args-json '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'
```

3. Build bundle (GRIB → bundle), if you need it:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --target-sfc-grib /path/hres_target_sfc.grib \
  --target-pl-grib  /path/hres_target_pl.grib \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

4. Build prediction-only bundle when target GRIBs are unavailable:
```bash
python -m manual_inference.prediction.predict build-bundle \
  --lres-sfc-grib /path/lres_sfc.grib \
  --lres-pl-grib  /path/lres_pl.grib \
  --hres-grib     /path/hres_static.grib \
  --allow-missing-target-unsafe \
  --out /home/ecm5702/hpcperm/data/input_data/o96/<bundle>.nc
```

**Notes**
- This workflow is **xarray-first**. Use MARS → xarray and build bundles from GRIB only when needed.
- The `--name-ckpt` can be a checkpoint name under the default root, or a full path.
