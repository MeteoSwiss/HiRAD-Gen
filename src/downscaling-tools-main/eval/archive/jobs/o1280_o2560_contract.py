from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass


@dataclass(frozen=True)
class O1280O2560Contract:
    lane: str
    description: str
    lres_sfc_channels: tuple[str, ...]
    lres_pl_channels: tuple[str, ...]
    hres_static_channels: tuple[str, ...]
    hres_dynamic_forcings: tuple[str, ...]
    target_sfc_channels: tuple[str, ...]
    target_pl_channels: tuple[str, ...]
    output_weather_states: tuple[str, ...]
    plot_weather_states: tuple[str, ...]
    spectra_weather_states: tuple[str, ...]
    output_weather_state_mode: str
    slim_output: bool
    required_gpu_ranks: int
    num_gpus_per_model: int
    cfgrib_filters: dict[str, dict[str, str]]


O1280_O2560_DESTINE_CONTRACT = O1280O2560Contract(
    lane="o1280_o2560",
    description=(
        "DestinE o1280->o2560 surface-only manual-inference contract for the "
        "current imported checkpoint family. Low-res inputs and targets use "
        "2t/10u/10v/msl; no PL outputs are part of the strict proof route."
    ),
    lres_sfc_channels=("10u", "10v", "2t", "msl"),
    lres_pl_channels=(),
    hres_static_channels=("z", "lsm"),
    hres_dynamic_forcings=(
        "cos_julian_day",
        "cos_latitude",
        "cos_local_time",
        "cos_longitude",
        "sin_julian_day",
        "sin_latitude",
        "sin_local_time",
        "sin_longitude",
    ),
    target_sfc_channels=("10u", "10v", "2t", "msl"),
    target_pl_channels=(),
    output_weather_states=("10u", "10v", "2t", "msl"),
    plot_weather_states=("10u", "10v", "2t", "msl"),
    spectra_weather_states=("10u", "10v", "2t", "msl"),
    output_weather_state_mode="all",
    slim_output=True,
    required_gpu_ranks=4,
    num_gpus_per_model=4,
    cfgrib_filters={
        "lres_surface": {"typeOfLevel": "surface"},
        "lres_pressure_levels": {"typeOfLevel": "isobaricInhPa"},
    },
)


def contract_to_dict(contract: O1280O2560Contract = O1280_O2560_DESTINE_CONTRACT) -> dict[str, object]:
    payload = asdict(contract)
    payload.update(
        {
            "lres_sfc_channels_csv": ",".join(contract.lres_sfc_channels),
            "lres_pl_channels_csv": ",".join(contract.lres_pl_channels),
            "hres_static_channels_csv": ",".join(contract.hres_static_channels),
            "hres_dynamic_forcings_csv": ",".join(contract.hres_dynamic_forcings),
            "target_sfc_channels_csv": ",".join(contract.target_sfc_channels),
            "target_pl_channels_csv": ",".join(contract.target_pl_channels),
            "output_weather_states_csv": ",".join(contract.output_weather_states),
            "plot_weather_states_csv": ",".join(contract.plot_weather_states),
            "spectra_weather_states_csv": ",".join(contract.spectra_weather_states),
        }
    )
    return payload
