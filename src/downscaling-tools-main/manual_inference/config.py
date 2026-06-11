"""Shared constants for the manual_inference package."""
from __future__ import annotations

DEFAULT_CONSTANT_FORCINGS_NPZ = "/home/ecm5702/hpcperm/data/o320-forcings.npz"
FALLBACK_CONSTANT_FORCINGS_NPZ = (
    "/home/ecm5702/hpcperm/data/o48-forcings.npz",
    "/home/ecm5702/hpcperm/data/o1280-forcings.npz",
    "/home/ecm5702/hpcperm/data/forcings_for_anemoi_inference/forcings_o320.npz",
    "/home/ecm5702/hpcperm/data/forcings_for_anemoi_inference/forcings_o1280.npz",
)

SFC_TO_CFGRIB = {
    "10u": "u10",
    "10v": "v10",
    "2d": "d2m",
    "2t": "t2m",
    "cp": "cp",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "tcw": "tcw",
    "tp": "tp",
}

CFGRIB_TO_SFC = {v: k for k, v in SFC_TO_CFGRIB.items()}

OPTIONAL_ZERO_LRES_SFC_VARS = frozenset(
    {
        "cp",
        "hcc",
        "lcc",
        "mcc",
        "ssrd",
        "strd",
        "tcc",
        "tp",
    }
)

BUNDLE_IMPLICIT_HRES_FEATURES = frozenset(
    {
        "z",
        "lsm",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "insolation",
    }
)

DEFAULT_LRES_SFC_CHANNELS = (
    "10u",
    "10v",
    "2d",
    "2t",
    "cp",
    "hcc",
    "lcc",
    "mcc",
    "msl",
    "skt",
    "sp",
    "ssrd",
    "strd",
    "tcc",
    "tcw",
    "tp",
)
DEFAULT_LRES_PL_CHANNELS = ("q", "t", "u", "v", "w", "z")
DEFAULT_TARGET_SFC_CHANNELS = ("10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "tp")
DEFAULT_TARGET_PL_CHANNELS = ("q", "t", "u", "v", "w", "z")

DEFAULT_EXTRA_ARGS_JSON = (
    '{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'
)

DATASET_PATH_REWRITE_PREFIXES = (
    ("/leonardo_work/DestE_340_25/ai-ml/datasets///", "/home/mlx/ai-ml/datasets/"),
    ("/leonardo_work/DestE_340_25/ai-ml/datasets/", "/home/mlx/ai-ml/datasets/"),
    ("/e/data1/jureap-data/ai-ml/datasets///", "/home/mlx/ai-ml/datasets/"),
    ("/e/data1/jureap-data/ai-ml/datasets/", "/home/mlx/ai-ml/datasets/"),
    ("/e/home/jusers/dumontlebrazidec1/jupiter/gkpdm/datasets///", "/home/mlx/ai-ml/datasets/"),
    ("/e/home/jusers/dumontlebrazidec1/jupiter/gkpdm/datasets/", "/home/mlx/ai-ml/datasets/"),
    ("/e/home/jusers/dumontlebrazidec1/jupiter/dev/.runtime_datasets/o1280_370523//", "/home/mlx/ai-ml/datasets/"),
    ("/e/home/jusers/dumontlebrazidec1/jupiter/dev/.runtime_datasets/o1280_370523/", "/home/mlx/ai-ml/datasets/"),
)

DEFAULT_CKPT_ROOT = "/home/ecm5702/scratch/aifs/checkpoint"

DEFAULT_EXPERIMENTS_DIR = "/home/ecm5702/hpcperm/experiments"
