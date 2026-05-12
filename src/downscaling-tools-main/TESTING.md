# Testing guide

## Test layout
- `manual_inference/tests`: inference and input-construction tests (CPU + GPU).
- `eval/tests`: eval module tests (CPU).
- `eval/jobs/tests`: job-script and orchestration tests (CPU).

## Runtime guardrails
- A global pytest hook enforces a hard max of 30 minutes per test.
- GPU tests are tagged with `@pytest.mark.gpu`.
- GPU tests are skipped unless `--run-gpu` is provided.

## Commands
- CPU-only suite:
  - `python -m pytest -m "not gpu"`
- GPU suite:
  - `python -m pytest -m gpu --run-gpu`
- GPU debug mode:
  - `python -m pytest -m gpu --run-gpu --gpu-debug`

## 30-minute GPU debug allocation (SLURM example)
- Request one GPU for up to 30 minutes, then run GPU tests in debug mode:
  - `srun --time=00:30:00 --gpus=1 python -m pytest -m gpu --run-gpu --gpu-debug`
