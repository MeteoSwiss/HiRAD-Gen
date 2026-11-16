# Golden tests: behaviour of the production checkpoints

These tests freeze what the current code does with every production checkpoint, so that a
refactor can be shown to change nothing. They run the real entry points (`generate.py`,
`train.py`) with recording hooks; nothing in `src/` is modified to capture.

They need a GPU node and the data stores on `/capstor`, so they are not part of the unit-test
run (`pytest tests`). Run them by hand (or on a schedule) through SLURM.

## What is captured

For every setup in [`registry.yaml`](registry.yaml) (one per regression + diffusion
checkpoint pair):

| Stage | Captured |
|---|---|
| `generate` | dataset fingerprint (channels, stats, sample times, `previous_index`, interpolation and regrid probes, raw and normalized samples); per step: preprocessed network inputs, the exact first-conv input of each U-Net (incl. positional/lead-time embeddings), first network outputs, seeds, generated ensemble; plus every torch and GRIB file `generate.py` writes |
| `train_regression`, `train_diffusion` | training-dataset fingerprint; warm start from the production weights (must load without missing/unexpected/mismatched keys); per step: preprocessed batch, network inputs, loss, noise levels, patch positions, learning rate, gradient norms; the saved `model_args.json` and state-dict layout |

Small arrays are stored in full; large tensors as a fingerprint (SHA-256 of the logical
float32 values, per-channel statistics, values at fixed sampled positions).

## Two GroupNorm modes

Every setup is captured twice, with the same production weights:

| Mode | GroupNorm | Comparison | Purpose |
|---|---|---|---|
| `torch` | PyTorch (`use_apex_gn=False`; apex checkpoints are converted on load), strict deterministic algorithms | exact: every record and output file bitwise identical | catches any change, however small |
| `apex` | apex, as in production | per-channel tolerances | keeps the production path covered |

The apex kernels accumulate GroupNorm statistics with float atomics, so two apex runs on the
same input differ in the last bits; the network and sampler amplify that to ~1e-4 of a
channel's std typically and up to ~1e-2 at single pixels. The PyTorch implementation sums in a
fixed order, so a torch capture repeats bitwise on the same GPU type and software stack (a new
container or GPU type needs new torch references).

## Workflow

Capture (one array task per setup, from the repo root; `GROUP_NORM=torch` or `apex`):

```bash
OUT=/capstor/store/cscs/pasc/c38/hirad_golden/<label> GROUP_NORM=torch \
  sbatch -A <account> --array=0-9 tests/golden/run_capture.sh
```

`SETUPS="era_real_cv1 era_cosmo"` and `STAGES="generate"` restrict the run (adjust `--array`).

Compare a new capture against the references of the same mode (the mode is read from each
capture's `manifest.json`; mixing modes is an error):

```bash
srun ... python -m tests.golden.compare --ref <refs> --new <new capture> [--setup era_real_cv1]
```

Tolerance policy: everything up to the network input (dataset, normalization, interpolation,
regridding, conditioning assembly, seeds, patch positions) is bitwise deterministic and must
be identical in both modes. In `torch` mode everything else must be identical too. In `apex`
mode values computed from network outputs are compared per channel, relative to the channel's
own variability, with two bounds: pointwise (worst value vs. channel range) and aggregate (RMS
error vs. channel std, per-channel statistics, scalar sums). Defaults
(`--{generate,train}-{pointwise,aggregate}`) are calibrated against an independent re-capture
of the baseline and checked against negative controls; the numbers are in `compare.py` and in
`NOTES.md` next to the references. A deliberate numerical change (e.g. precision) gets new
references under a new label, with the reason recorded next to them.

Current references (`/capstor/store/cscs/pasc/c38/hirad_golden/`):

| Mode | References | Notes |
|---|---|---|
| `torch` | `baseline-8bfd985-torchgn/` | 8bfd985 + the GroupNorm-switch `src/` patch stored there; GH200, `ci/edf/modulus_env.toml` |
| `apex` | `baseline-8bfd985/` | 8bfd985 |

## Updating the hooks

Record names are semantic (`step1/diffusion/unet_input`, `dataset/output_mean`, ...). When a
refactor renames an interface the hooks wrap (`Generator.generate`,
`TrainingManagerCorrDiff.load_and_preprocess_batch`, ...), update [`hooks.py`](hooks.py) so it
produces the same records; never regenerate references to make a comparison pass.

## Registry

Each setup lists its checkpoints, the entry config and Hydra overrides it is run with, the
test times, and the provenance of that mapping (the original training configs were lost to
the scratch purge). The registry is also the source for persisting a data specification next
to each checkpoint. A `generate` stage can set its own `num_ensembles` / `seed_batch_size`
(default: `common`); `era_real_tc` uses this so its autoregressive rollout runs with several
seed batches and across a gap in the times.

The captured code is the `hirad` package first on `PYTHONPATH` (recorded as `hirad_source` in
the manifest), with that package's Hydra configs. Normally this is the repo's `src/`; to capture
an older commit with the current harness, put an export of it first
(`git archive <commit> src | tar -x -C <dir>`, then `PYTHONPATH=<dir>/src:$PWD`).
