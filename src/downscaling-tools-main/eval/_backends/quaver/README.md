# Quaver Verification Scripts

This directory contains the FDB-backed verification scripts invoked by `eval/jobs/launch_full_eval_suite.sh --expver <EXPVER>`. They are not part of the `python -m eval.cli evaluate` framework and do not consume the `predictions_*.nc` outputs of `eval/predict/`.

## Files

- `q_compute_probabilistic.py` - driver script. Run with `quaver q_compute_probabilistic.py --expver <X> --nmem <N> ...` after `module load quaver`. Reads forecasts and analysis from FDB and writes probabilistic scores back to FDB.
- `q_plot_pl.py`, `q_plot_sfc.py` - plotting drivers. Read scores from FDB and emit `quaver.pdf`. Also run under the `quaver` binary.
- `compute_quaver.sh` - example sbatch wrapper.
- `__init__.py` - empty package marker; not imported by the evaluator framework.

## When To Use

Use this path when you have an upstream MARS expver from prepml / anemoi-inference and want operational-style verification against FDB analysis and surface observations. Do not use this path against the downscaling-tools `predictions_*.nc` ensemble outputs; there is no evaluator wrapper for that, and quaver does not natively read NetCDF ensemble files.

## Constraints

- Requires `module load quaver` on ECMWF AC/AG environments.
- Requires FDB read access.
- Output is a single `quaver.pdf`, not the `evaluators/<name>/{metrics.json,plots/}` shape used by the eval framework.

## Canonical Entry Point

```bash
/home/ecm5702/dev/downscaling-tools/eval/jobs/launch_full_eval_suite.sh \
    --expver <EXPVER> [--quaver-first-date YYYYMMDD] [--quaver-last-date YYYYMMDD] \
    [--quaver-nmem N] [--hres-grid O320|...]
```
