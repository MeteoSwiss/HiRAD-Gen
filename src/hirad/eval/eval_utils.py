import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple

from hirad.datasets import get_channels_from_strings, known_datasets
from hirad.utils.function_utils import get_time_from_range


def load_generation_setup(cfg: dict) -> Tuple[Path, dict, list]:
    """Validate ``cfg['inference_output_dir']``, load its generation config, and resolve times.

    Returns ``(generation_dir, gen_cfg, times)``. Raises :class:`ValueError` with a
    descriptive message when validation fails.
    """
    generation_dir = cfg.get("inference_output_dir")
    if generation_dir is None:
        raise ValueError("No inference_output_dir specified in config.")

    generation_dir = Path(generation_dir)
    if not generation_dir.is_dir():
        raise ValueError(
            f"Inference output directory {generation_dir} does not exist or is not a directory."
        )

    generation_config_path = find_generation_config(str(generation_dir))
    if generation_config_path is None:
        raise ValueError(f"No generation config file found in {generation_dir}.")

    with open(generation_config_path, "r") as f:
        gen_cfg = yaml.safe_load(f)

    times = resolve_times(cfg, gen_cfg)
    if times is None:
        raise ValueError(
            "No times, times_range, or times_ranges specified in config or generation config."
        )

    return generation_dir, gen_cfg, times


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


def find_generation_config(generation_dir: str) -> Optional[Path]:
    """Return the first `.hydra/config.yaml` found under *generation_dir*, or None."""
    return min(Path(generation_dir).glob("**/.hydra/config.yaml"), default=None)


def resolve_ts_dir(out_root: Path, ts: str) -> Path:
    """Return the directory under *out_root* that contains the timestamp folder *ts*."""
    if (out_root / ts).is_dir():
        return out_root
    matches = [p.parent for p in out_root.glob(f"*/{ts}") if p.is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Timestamp directory {ts} not found under {out_root}")


def load_timestep_tensor(out_root: Path, ts: str, name: str):
    """Load ``<out_root>/<ts>/<ts>-<name>`` as a torch tensor (``weights_only=False``)."""
    import torch
    return torch.load(
        resolve_ts_dir(out_root, ts) / ts / f"{ts}-{name}",
        weights_only=False,
    )


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
            help="One or more timesteps to plot (format: YYYYMMDD-HHMM). "
                 "Overrides times/times_range/times_ranges in the config.",
        )
    args = parser.parse_args()

    with open(args.config_name, "r") as f:
        cfg = yaml.safe_load(f)

    if allow_times and getattr(args, "times", None):
        for k in ("times", "times_range", "times_ranges"):
            cfg.pop(k, None)
        cfg["times"] = args.times

    return cfg


def resolve_times(cfg: dict, gen_cfg: dict, time_format: str = "%Y%m%d-%H%M") -> Optional[list]:
    """Resolve the list of timestep strings from eval or generation config.

    Priority order (both in eval cfg and generation cfg fallback):
      1. ``times_ranges`` – list of [start, end, step] ranges, concatenated.
      2. ``times_range``  – single [start, end, step] range.
      3. ``times``        – explicit list of strings.

    Returns ``None`` when no time specification is found in either config.
    """
    def _from_cfg(source: dict) -> Optional[list]:
        if source.get("times_ranges"):
            times = []
            for tr in source["times_ranges"]:
                times.extend(get_time_from_range(tr, time_format=time_format))
            return times
        if source.get("times_range"):
            return get_time_from_range(source["times_range"], time_format=time_format)
        if source.get("times"):
            return source["times"]
        return None

    return _from_cfg(cfg) or _from_cfg(gen_cfg.get("generation", {}))


def percentiles_from_histogram(hist_counts, bin_edges, percentiles_dict):
    """
    Estimate percentiles from a pre-computed histogram using linear interpolation
    on the cumulative distribution.
    
    Parameters
    ----------
    hist_counts : np.ndarray
        Raw (unnormalized) histogram counts per bin.
    bin_edges : np.ndarray
        Bin edges (length = len(hist_counts) + 1).
    percentiles_dict : dict
        Mapping of label -> fractional percentile, e.g. {99: 0.99, 99.9: 0.999}.
    
    Returns
    -------
    dict
        Mapping of label -> estimated percentile value.
    """
    cumulative = np.cumsum(hist_counts)
    total = cumulative[-1]
    if total == 0:
        return {key: np.nan for key in percentiles_dict}
    
    cdf = cumulative / total  # CDF at upper bin edges
    
    results = {}
    for key, p in percentiles_dict.items():
        # Find the bin where CDF crosses p
        idx = np.searchsorted(cdf, p)
        if idx >= len(cdf):
            # Beyond last bin — return upper edge
            results[key] = bin_edges[-1]
        elif idx == 0:
            # Within first bin — linearly interpolate from 0
            frac = p / cdf[0] if cdf[0] > 0 else 0.0
            results[key] = bin_edges[0] + frac * (bin_edges[1] - bin_edges[0])
        else:
            # Linearly interpolate within the bin
            cdf_low = cdf[idx - 1]
            cdf_high = cdf[idx]
            frac = (p - cdf_low) / (cdf_high - cdf_low) if (cdf_high - cdf_low) > 0 else 0.0
            results[key] = bin_edges[idx] + frac * (bin_edges[idx + 1] - bin_edges[idx])
    
    return results