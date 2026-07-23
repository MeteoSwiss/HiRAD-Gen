"""Measure the true statistics of the normalized HR target for the DiT loss.

Two things:
  1. Per-channel std of normalize_output(target) — the effective EDM sigma_data.
  2. Raw box-cox(tp) mean/std, compared to the config's transform_output stats
     (tp: mean=-3.8121, std=0.3346). If these are wrong, tp is mis-normalized.

Streams over the TRAINING period (2005-2020) with float64 accumulation so the
heavy-tailed precip std is estimated robustly (not the 64-sample/2023 first pass).

Run (CPU is fine, ~10-20 min):
  srun -A c38 --partition=debug --nodes=1 --ntasks=1 --gpus-per-node=1 --time=00:25:00 \
    --environment=/users/pstamenk/HiRAD-Gen/ci/edf/modulus_env.toml bash -c \
    'cd /users/pstamenk/HiRAD-Gen && source ../hirad_new_env/bin/activate && \
     python src/hirad/eval/measure_sigma_data.py'
"""
import numpy as np
import torch
import yaml

from hirad.datasets import known_datasets

N = 300           # timesteps sampled across the training record
LAMBDA = 0.25     # tp box-cox lambda (tp-box_cox_025)
CFG_TP_MEAN, CFG_TP_STD = -3.8121187083242556, 0.3345851858215482

cfg = yaml.safe_load(open("src/hirad/conf/dataset/anemoi_era_real_inference.yaml"))
cfg["start_date"], cfg["end_date"] = "2005-01-02", "2020-12-31"   # training period
ds = known_datasets[cfg["type"]](**cfg)
names = [getattr(c, "name", str(c)) for c in ds.output_channels()]
C = len(names)
tp = names.index("tp")
print(f"channels: {names} | dataset len: {len(ds)} | sampling {N} timesteps")

s = np.zeros(C); s2 = np.zeros(C); cnt = 0            # normalized, per channel
bsum = bsq = bcnt = 0.0                               # raw box-cox(tp)
idxs = np.linspace(0, len(ds) - 1, N).astype(int)
for k, i in enumerate(idxs):
    raw = torch.as_tensor(np.asarray(ds[int(i)][0])).double()   # (C, S) physical
    bc = ds.box_cox_transform(raw[tp].clone(), LAMBDA).numpy().ravel()
    bsum += bc.sum(); bsq += (bc ** 2).sum(); bcnt += bc.size
    norm = ds.normalize_output(raw[None].clone())[0].numpy().reshape(C, -1)
    s += norm.sum(1); s2 += (norm ** 2).sum(1); cnt += norm.shape[1]
    if k % 50 == 0 or k == N - 1:
        print(f"  {k+1}/{N}")

mean = s / cnt; std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
bmean = bsum / bcnt; bstd = float(np.sqrt(max(bsq / bcnt - bmean ** 2, 0)))

print("\n--- normalized-output std per channel (effective sigma_data) ---")
for nm, m, sd in zip(names, mean, std):
    print(f"  {nm:>6}: std = {sd:.4f}   (mean = {m:+.4f})")

print("\n--- raw box-cox(tp) stats vs config ---")
print(f"  measured : mean = {bmean:.4f}   std = {bstd:.4f}")
print(f"  config   : mean = {CFG_TP_MEAN:.4f}   std = {CFG_TP_STD:.4f}")
print(f"  => with CORRECT stats, normalized tp std -> 1.0 (currently {bstd/CFG_TP_STD:.3f})")
