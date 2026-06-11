# eval/predict — Modular Prediction Generation

Replaces `generate_predictions_25_files.py` with clean module boundaries, proper date handling, and schema validation.

## Quick Start

```bash
# From downscaling-tools root:
python -m eval.predict.main \
  --input-root /path/to/bundles \
  --out-dir /path/to/output \
  --name-ckpt /path/to/checkpoint.ckpt \
  --dates 20230826,20230827,20230828,20230829,20230830 \
  --steps 24,48,72,96,120 \
  --members 1,2,3,4,5,6,7,8,9,10
```

## Architecture

```
eval/predict/
├── types.py            — BundleKey, PredictionConfig, PredictionMetadata
├── bundle_manager.py   — Bundle discovery & filename parsing
├── model_loader.py     — Checkpoint loading & device setup
├── dataset_builder.py  — xr.Dataset assembly with real dates & CF metadata
├── inference_engine.py — Core prediction loop with memory management
├── distributed_io.py   — Rank-0 file writing coordination
├── output_writer.py    — NC file writing with schema validation
├── main.py             — CLI entry point & orchestration
└── tests/              — Unit tests
```

## What's Fixed

### Date Handling (Critical Bug)
The old script passed `dates=None` to `build_predictions_dataset()`, which created `date=0` in output files. Plotters interpreted this as Unix epoch → "1970-01-01" in titles.

**Now:** Real init dates are extracted from bundle filenames and written as `datetime64[ns]` with CF-compliant attributes.

### Module Boundaries
Each concern has its own module:
- **Bundle discovery** — regex parsing, file system traversal
- **Model loading** — checkpoint resolution, device setup, distributed init
- **Dataset building** — array assembly, metadata, schema validation
- **Inference** — prediction loop, memory management, member stacking
- **Distributed I/O** — rank-0 marker files, polling, timeout handling
- **Output writing** — combines dataset building + distributed I/O + file writing

### Schema Validation
`validate_predictions_dataset(ds)` checks output files for:
- Required variables (x, y_pred, lon/lat)
- Required coordinates (sample, ensemble_member, weather_state)
- Valid date metadata (not epoch placeholder)

## Output Format

Files: `predictions_{YYYYMMDD}_step{NNN}.nc`

This naming convention is required — TC eval, spectra eval, and regional plotting all discover files with `re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")`.

## Migration from generate_predictions_25_files.py

The CLI interface is backwards-compatible. Replace:
```bash
python eval/jobs/generate_predictions_25_files.py --input-root ... --out-dir ...
```
with:
```bash
python -m eval.predict.main --input-root ... --out-dir ...
```

All arguments are preserved. The only behavioral change is correct date metadata in output files.
