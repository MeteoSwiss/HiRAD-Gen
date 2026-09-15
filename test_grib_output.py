#!/usr/bin/env python3
"""
Test convert_torch_to_grib by loading previously-saved torch output and writing GRIB.

Usage:
    python test_grib_output.py [time_step]

    time_step defaults to '20230115-0000'
"""

import os
import sys
from omegaconf import OmegaConf

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.utils.inference_utils import convert_torch_to_grib

TIME_STEP = sys.argv[1] if len(sys.argv) > 1 else '20230115-0000'

TORCH_OUT_DIR = './outputs/generation/generation_era_real_test/outputs/evaluation/era_real_test'
GRIB_TEMPLATE_PATH = os.path.expanduser('~/evalml/resources/inference/templates')
DATASET_CFG = 'src/hirad/conf/dataset/anemoi_era_real_inference.yaml'

# Load dataset for channel metadata
dataset_cfg = OmegaConf.to_container(OmegaConf.load(DATASET_CFG))
dataset, _ = get_dataset_and_sampler_inference(dataset_cfg=dataset_cfg, times=[TIME_STEP])

print(f'Loading torch files from {TORCH_OUT_DIR} for {TIME_STEP} ...')
grib_savedir = convert_torch_to_grib(TORCH_OUT_DIR, TIME_STEP, dataset, GRIB_TEMPLATE_PATH)

print(f'\nOutput files in {grib_savedir}:')
for f in sorted(os.listdir(grib_savedir)):
    path = os.path.join(grib_savedir, f)
    print(f'  {f}  ({os.path.getsize(path):,} bytes)')

# Quick sanity check with earthkit
try:
    import earthkit.data as ekd
    for f in sorted(os.listdir(grib_savedir)):
        ds = ekd.from_source('file', os.path.join(grib_savedir, f))
        print(f'\n{f}:')
        print(ds.ls().to_string(index=False))
except Exception as e:
    print(f'\n(earthkit check skipped: {e})')
