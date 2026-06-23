import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import yaml

from hirad.datasets import get_channels_from_strings, get_strings_from_channels, known_datasets
from hirad.utils.function_utils import get_time_from_range


@dataclass
class GridConfig:
    lat: np.ndarray
    lon: np.ndarray
    height: int
    width: int
    relax_zone: int


DEFAULT_GRID_CONFIG = GridConfig(
    lat=np.arange(-4.42, 3.36 + 0.02, 0.02),
    lon=np.arange(-6.82, 4.80 + 0.02, 0.02),
    height=352,
    width=544,
    relax_zone=19,
)


def grid_cfg_from_cfg(cfg) -> GridConfig:
    """Build a :class:`GridConfig` from ``lat_*``/``lon_*``/``height``/``width``/``relax_zone`` fields of *cfg*."""
    return GridConfig(
        lat=np.arange(cfg.get("lat_start"), cfg.get("lat_end") + cfg.get("lat_step"), cfg.get("lat_step")),
        lon=np.arange(cfg.get("lon_start"), cfg.get("lon_end") + cfg.get("lon_step"), cfg.get("lon_step")),
        height=cfg.get("height"),
        width=cfg.get("width"),
        relax_zone=cfg.get("relax_zone"),
    )


def precip_conv_factor(cfg: dict) -> float:
    """Return the single factor converting stored precipitation to mm/h.

    ERA5 precipitation is an hourly accumulated depth in metres, so the only
    physical conversion is metres -> millimetres (x1000), yielding mm/h. This is
    the *one* precipitation unit used throughout evaluation: per-day and multi-day
    totals (Rx1day, Rx5day, CDD/CWD, ...) are obtained by summing mm/h values over
    time, never via a separate scaling factor.
    """
    return float(cfg.get("precip_conv_factor", 1000.0))


def load_land_sea_mask(path, height=352, width=544):
    """Load and return a land-sea mask as xarray DataArray."""
    lsm_data = np.load(path).reshape(height, width)
    return xr.DataArray(
        np.where(lsm_data >= 0.5, 1.0, np.nan),
        dims=['lat', 'lon'],
        coords={"lat": np.arange(height), "lon": np.arange(width)},
    )


def concat_and_group_diurnal(list_of_da, is_member=False, scale=1.0):
    """Concatenate DataArrays along ``time`` and compute diurnal mean (and member std)."""
    da = xr.concat(list_of_da, dim="time")
    if is_member:
        mean = da.groupby("time.hour").mean(dim="time").mean(dim="member") * scale
        std = da.std(dim="member").groupby("time.hour").mean(dim="time") * scale
    else:
        mean = da.groupby("time.hour").mean(dim="time") * scale
        std = None
    return mean, std


def percentiles_from_histogram(hist_counts, bin_edges, percentiles_dict):
    """Estimate percentiles from a histogram via linear interpolation on the CDF.

    Parameters
    ----------
    hist_counts : np.ndarray
        Raw (unnormalized) histogram counts per bin.
    bin_edges : np.ndarray
        Bin edges (length = ``len(hist_counts) + 1``).
    percentiles_dict : dict
        Mapping ``label -> fractional percentile`` (e.g. ``{99: 0.99}``).

    Returns
    -------
    dict
        Mapping ``label -> estimated value``.
    """
    cumulative = np.cumsum(hist_counts)
    total = cumulative[-1]
    if total == 0:
        return {key: np.nan for key in percentiles_dict}

    cdf = cumulative / total  # CDF at upper bin edges
    results = {}
    for key, p in percentiles_dict.items():
        idx = np.searchsorted(cdf, p)
        if idx >= len(cdf):
            results[key] = bin_edges[-1]
        elif idx == 0:
            frac = p / cdf[0] if cdf[0] > 0 else 0.0
            results[key] = bin_edges[0] + frac * (bin_edges[1] - bin_edges[0])
        else:
            cdf_low, cdf_high = cdf[idx - 1], cdf[idx]
            frac = (p - cdf_low) / (cdf_high - cdf_low) if (cdf_high - cdf_low) > 0 else 0.0
            results[key] = bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx])
    return results


def load_generation_setup(cfg: dict) -> Tuple[Path, dict, list]:
    """Validate ``cfg['inference_output_dir']``, load its generation config, and resolve times.

    Returns ``(generation_dir, gen_cfg, times)``. Raises :class:`ValueError` on failure.
    """
    generation_dir = cfg.get("inference_output_dir")
    if generation_dir is None:
        raise ValueError("No inference_output_dir specified in config.")

    generation_dir = Path(generation_dir)
    if not generation_dir.is_dir():
        raise ValueError(f"Inference output directory {generation_dir} does not exist or is not a directory.")

    generation_config_path = min(generation_dir.glob("**/.hydra/config.yaml"), default=None)
    if generation_config_path is None:
        raise ValueError(f"No generation config file found in {generation_dir}.")

    with open(generation_config_path, "r") as f:
        gen_cfg = yaml.safe_load(f)

    times = _resolve_times(cfg, gen_cfg)
    if times is None:
        raise ValueError("No times, times_range, or times_ranges specified in config or generation config.")

    return generation_dir, gen_cfg, times


def _resolve_times(cfg: dict, gen_cfg: dict, time_format: str = "%Y%m%d-%H%M") -> Optional[list]:
    """Resolve timestep strings from eval cfg, falling back to ``gen_cfg['generation']``.

    Priority (in each source): ``times_ranges`` > ``times_range`` > ``times``.
    """
    def _from(source: dict) -> Optional[list]:
        if source.get("times_ranges"):
            return [t for tr in source["times_ranges"] for t in get_time_from_range(tr, time_format=time_format)]
        if source.get("times_range"):
            return get_time_from_range(source["times_range"], time_format=time_format)
        return source.get("times")

    return _from(cfg) or _from(gen_cfg.get("generation", {}))


def resolve_io_channels(gen_cfg: dict) -> Tuple[list, list]:
    """Resolve ``(input_channels, output_channels)`` from a generation config.

    Uses ``input_channel_names`` / ``output_channels_names`` from ``gen_cfg['dataset']``
    when available; otherwise instantiates the dataset and queries it.
    """
    dataset_cfg = gen_cfg.get("dataset", {})
    input_channels = get_channels_from_strings(dataset_cfg.get("input_channel_names", []))
    output_channels = get_channels_from_strings(dataset_cfg.get("output_channels_names", []))
    if not input_channels or not output_channels:
        dataset = known_datasets[dataset_cfg.get("type")](**dataset_cfg)
        input_channels = dataset.input_channels()
        output_channels = dataset.output_channels()
    return input_channels, output_channels


def get_channel_indices(gen_cfg: dict, channels=None) -> dict:
    """Return ``{'input': {name: idx}, 'output': {name: idx}}`` from a generation config.

    When *channels* is given, the mappings are filtered to only those names.
    """
    input_channels, output_channels = resolve_io_channels(gen_cfg)
    in_ch = {get_strings_from_channels(c): i for i, c in enumerate(input_channels)}
    out_ch = {get_strings_from_channels(c): i for i, c in enumerate(output_channels)}
    if channels is None:
        return {'input': in_ch, 'output': out_ch}
    return {
        'input': {ch: in_ch[ch] for ch in channels if ch in in_ch},
        'output': {ch: out_ch[ch] for ch in channels if ch in out_ch},
    }


def make_percentile_values(per_decade: int = 20) -> np.ndarray:
    """Percentiles sampled equidistantly on the logit (log-exceedance) axis."""
    tail = np.logspace(-3, 1, 4 * per_decade + 1)   # 0.001 ... 10
    lower = tail                                     # low tail:  0.001 ... 10
    upper = 100.0 - tail[::-1]                       # high tail: 90 ... 99.999
    center = np.linspace(10.0, 90.0, 2 * per_decade + 1)
    return np.unique(np.concatenate([lower, center, upper]))


def resolve_ts_dir(out_root: Path, ts: str) -> Path:
    """Return the directory under *out_root* that contains the timestamp folder *ts*."""
    if (out_root / ts).is_dir():
        return out_root
    matches = [p.parent for p in out_root.glob(f"*/{ts}") if p.is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Timestamp directory {ts} not found under {out_root}")


def signed_circular_difference(prediction: np.ndarray, target: np.ndarray, period: float = 360.0) -> np.ndarray:
    """Return signed wrapped difference on a circular domain.

    For angles in degrees, this yields values in [-180, 180).
    """
    half_period = period / 2.0
    return ((prediction - target + half_period) % period) - half_period



def parse_eval_cli(allow_times: bool = False) -> dict:
    """Parse standard eval CLI args (``--config-name``) and return the loaded YAML config.

    When ``allow_times=True``, also accepts ``--times YYYYMMDD-HHMM ...`` to override
    ``times`` / ``times_range`` / ``times_ranges`` in the config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", help="Path to YAML config file for evaluation.")
    if allow_times:
        parser.add_argument(
            "--times", nargs="+",
            help="One or more timesteps (YYYYMMDD-HHMM) overriding times/times_range/times_ranges.",
        )
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    if allow_times and getattr(args, "times", None):
        for k in ("times", "times_range", "times_ranges"):
            cfg.pop(k, None)
        cfg["times"] = args.times

    return cfg
