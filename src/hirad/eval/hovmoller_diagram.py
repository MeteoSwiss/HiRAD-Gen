"""
Hovmöller Diagrams for hourly Wind Speed

Creates longitude-time plots along a latitude slice through Switzerland with
bandpass filtering to isolate different timescales (diurnal, synoptic, sub-seasonal).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import hydra
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from scipy import signal

from hirad.datasets import get_dataset_and_sampler_inference
from hirad.distributed import DistributedManager
from hirad.eval.plotting import get_channel_indices, LOG_INTERVAL, LON, RELAX_ZONE
from hirad.utils.function_utils import get_time_from_range


# Switzerland latitude range: approximately 45.8°N to 47.8°N (rotated coordinates)
# In the COSMO-2 rotated grid, Switzerland is around latitude index 200-250
LAT_INDEX = 220  # Approximate latitude cutting through central Switzerland
LAT_RANGE = (200, 250)  # Range for averaging


def bandpass_filter(
    data: xr.DataArray,
    low_period_hours: float,
    high_period_hours: float,
    time_dim: str = 'time',
    order: int = 5
) -> xr.DataArray:
    """Apply Butterworth bandpass filter using zero-phase filtering (sosfiltfilt)."""
    # Assume hourly sampling
    fs = 1.0  # 1/hour
    nyq = 0.5 * fs
    
    # Bandpass filter
    low_freq = 1.0 / low_period_hours
    high_freq = 1.0 / high_period_hours
    sos = signal.butter(order, [low_freq / nyq, high_freq / nyq], btype='band', output='sos')
    
    # Apply zero-phase filter
    filtered = xr.apply_ufunc(
        lambda x: signal.sosfiltfilt(sos, x, axis=0),
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    
    return filtered


def create_hovmoller_data(
    data_list: list,
    times: list,
    lat_index: int,
    lat_range: Optional[Tuple[int, int]] = None
) -> xr.DataArray:
    """Create Hovmöller data array (time, lon) from 2D spatial fields."""
    if lat_range is not None:
        # Average over latitude range
        lat_slice = slice(lat_range[0], lat_range[1])
        hovmoller_data = np.array([data[lat_slice, :].mean(axis=0) for data in data_list])
    else:
        # Extract single latitude
        hovmoller_data = np.array([data[lat_index, :] for data in data_list])
    
    # Create longitude coordinates (account for relaxation zone)
    lon_coords = LON[RELAX_ZONE : RELAX_ZONE + hovmoller_data.shape[1]]
    
    return xr.DataArray(
        hovmoller_data,
        dims=['time', 'lon'],
        coords={
            'time': times,
            'lon': lon_coords
        }
    )


def plot_hovmoller(
    data: xr.DataArray,
    filename: str,
    title: str = '',
    cbar_label: str = '',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    anomaly: bool = False
) -> None:
    """Create and save Hovmöller diagram (longitude-time plot)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # If anomaly plot, center at zero
    if anomaly and vmin is None and vmax is None:
        abs_max = np.abs(data.values).max()
        vmin, vmax = -abs_max, abs_max
    
    # Create mesh plot
    im = ax.pcolormesh(
        data.lon.values,
        data.time.values,
        data.values.T if data.values.shape[0] != len(data.time) else data.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='nearest'
    )
    
    # Format axes
    ax.set_xlabel('Longitude [degrees]', fontsize=12)
    ax.set_ylabel('Time', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    
    # Format time axis
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_generate")
def main(cfg: DictConfig):
    """Generate Hovmöller diagrams for wind speed with bandpass filtering."""
    DistributedManager.initialize()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load times
    times = get_time_from_range(cfg.generation.times_range, "%Y%m%d-%H%M")
    datetimes = [datetime.strptime(ts, "%Y%m%d-%H%M") for ts in times]
    logger.info(f"Generating Hovmöller diagrams for {len(times)} timesteps")
    
    # Dataset
    ds_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, _ = get_dataset_and_sampler_inference(
        ds_cfg, times, cfg.generation.get('has_lead_time', False)
    )
    
    # Output path
    out_root = Path(cfg.generation.io.output_path or './outputs')
    hovmoller_dir = out_root / 'hovmoller_diagrams'
    hovmoller_dir.mkdir(parents=True, exist_ok=True)
    
    # Get wind component channels
    indices = get_channel_indices(dataset)
    
    # Configure wind speed variable
    u10_out = indices['output'].get('10u')
    v10_out = indices['output'].get('10v')
    u10_in = indices['input'].get('10u', u10_out)
    v10_in = indices['input'].get('10v', v10_out)
    
    if u10_out is None or v10_out is None:
        logger.error("Wind components (10u, 10v) not found in dataset!")
        return
    
    var_config = {
        'out_ch': (u10_out, v10_out),
        'in_ch': (u10_in, v10_in),
        'label': '10m Wind Speed [m/s]',
        'convert': lambda u, v: np.hypot(u, v),
        'cmap': 'YlOrRd',
        'vmin': 0,
        'vmax': 15
    }
    
    # Bandpass filter configurations for different timescales
    filter_configs = {
        'full': {
            'low_period': None,
            'high_period': None,
            'label': 'Full Signal'
        },
        'diurnal': {
            'low_period': 24,    # hours
            'high_period': 2,   # hours
            'label': 'Diurnal (2-24h)'
        },
        'synoptic': {
            'low_period': 120,   # 5 days
            'high_period': 25,   # 1 day
            'label': 'Synoptic (2-5 days)'
        },
    }
    
    # Processing modes: data sources to compare
    modes = {
        'target': 'COSMO-2 Analysis',
        'baseline': 'ERA5',
        'regression-prediction': 'Regression Prediction'
    }
    
    # Process basic modes (target, baseline, regression-prediction)
    for mode, mode_label in modes.items():
        # Load all timesteps for this mode
        data_list = []
        try:
            for i, ts in enumerate(times):
                data = torch.load(out_root / ts / f"{ts}-{mode}", weights_only=False)
                
                # Extract wind components and compute speed
                u_ch, v_ch = (var_config['out_ch'] if mode == 'target' or mode.startswith('regression')
                             else var_config['in_ch'])
                field = var_config['convert'](data[u_ch], data[v_ch])
                
                data_list.append(field)
                
        except Exception as e:
            logger.warning(f"Mode {mode} not available, skipping: {e}")
            continue
        
        # Create Hovmöller data (average over latitude range)
        hovmoller_raw = create_hovmoller_data(
            data_list, datetimes,
            lat_index=LAT_INDEX,
            # lat_range=LAT_RANGE
        )
        
        # Apply filters and generate plots
        for filter_name, filter_cfg in filter_configs.items():
            # Apply bandpass filter (skip if no filter periods)
            if filter_cfg['low_period'] is None or filter_cfg['high_period'] is None:
                hovmoller_filtered = hovmoller_raw
                anomaly = False
            else:
                hovmoller_filtered = bandpass_filter(
                    hovmoller_raw,
                    low_period_hours=filter_cfg['low_period'],
                    high_period_hours=filter_cfg['high_period']
                )
                anomaly = True
            
            # Generate plot
            plot_title = (f"{mode_label}: Wind Speed\n"
                         f"{filter_cfg['label']} - Lat slice through Switzerland")
            
            filename = hovmoller_dir / f"{mode}_windspeed_{filter_name}"
            
            plot_hovmoller(
                hovmoller_filtered,
                str(filename),
                title=plot_title,
                cbar_label=var_config['label'] if not anomaly else f"{var_config['label']} Anomaly",
                vmin=var_config['vmin'] if not anomaly else None,
                vmax=var_config['vmax'] if not anomaly else None,
                cmap=var_config['cmap'],
                anomaly=anomaly
            )
    
    # Process diffusion model predictions (ensemble members)
    try:
        # Check predictions and get ensemble size
        test_data = torch.load(out_root / times[0] / f"{times[0]}-predictions", weights_only=False)
        n_members = test_data.shape[0]
        
        # Process each ensemble member (limit to 5 for memory)
        max_members = min(n_members, 5)
        for member_idx in range(max_members):
            # Load wind speed for this member
            data_list = []
            for i, ts in enumerate(times):
                pred_data = torch.load(out_root / ts / f"{ts}-predictions", weights_only=False)
                
                # Extract wind components and compute speed
                u_ch, v_ch = var_config['out_ch']
                field = var_config['convert'](
                    pred_data[member_idx, u_ch],
                    pred_data[member_idx, v_ch]
                )
                
                data_list.append(field)
            
            # Create Hovmöller data
            hovmoller_raw = create_hovmoller_data(
                data_list, datetimes,
                lat_index=LAT_INDEX,
                lat_range=LAT_RANGE
            )
            
            # Apply filters and generate plots
            for filter_name, filter_cfg in filter_configs.items():
                if filter_cfg['low_period'] is None or filter_cfg['high_period'] is None:
                    hovmoller_filtered = hovmoller_raw
                    anomaly = False
                else:
                    hovmoller_filtered = bandpass_filter(
                        hovmoller_raw,
                        low_period_hours=filter_cfg['low_period'],
                        high_period_hours=filter_cfg['high_period']
                    )
                    anomaly = True
                
                plot_title = (f"CorrDiff Member {member_idx+1}: Wind Speed\n"
                             f"{filter_cfg['label']} - Lat slice through Switzerland")
                
                filename = hovmoller_dir / f"prediction_member_{member_idx:02d}_windspeed_{filter_name}"
                
                plot_hovmoller(
                    hovmoller_filtered,
                    str(filename),
                    title=plot_title,
                    cbar_label=var_config['label'] if not anomaly else f"{var_config['label']} Anomaly",
                    vmin=var_config['vmin'] if not anomaly else None,
                    vmax=var_config['vmax'] if not anomaly else None,
                    cmap=var_config['cmap'],
                    anomaly=anomaly
                )
    
    except Exception as e:
        logger.warning(f"Predictions not available: {e}")
    
    logger.info(f"Hovmöller diagrams saved to {hovmoller_dir}")


if __name__ == '__main__':
    main()
