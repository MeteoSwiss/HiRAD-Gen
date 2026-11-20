"""Every entry config must compose and describe a runnable setup (no data access needed)."""

import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hirad.datasets import AnemoiDataset, known_datasets

# The YAML configs are not installed as package data, so read them from the checkout.
CONF_DIR = Path(__file__).resolve().parents[2] / "src" / "hirad" / "conf"
ENTRY_CONFIGS = sorted(
    p.stem for p in CONF_DIR.glob("*.yaml") if p.stem.startswith(("training_", "generate_"))
)
# Keys consumed by init_train_valid_datasets_from_config before the dataset is constructed.
SPLIT_KEYS = {"validation", "validation_start_date", "validation_end_date"}

# Broken today; strict xfail so that fixing or deleting one forces this list to be updated.
KNOWN_BROKEN = {
}


def compose_entry(name: str):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name=name)


def config_problems(cfg) -> list[str]:
    dataset = OmegaConf.to_container(cfg.dataset)
    dataset_type = dataset.get("type")
    if dataset_type not in known_datasets:
        return [f"dataset type {dataset_type} is not registered"]
    dataset_class = known_datasets[dataset_type]
    problems = []
    if not issubclass(dataset_class, AnemoiDataset):
        problems.append(f"{dataset_class.__name__} lacks the interface the training/inference loop uses")
    accepted = set(inspect.signature(dataset_class.__init__).parameters) - {"self"}
    unknown = sorted(set(dataset) - accepted - SPLIT_KEYS)
    if unknown:
        problems.append(f"dataset keys not accepted by {dataset_class.__name__}: {unknown}")

    model = cfg.get("model")
    if model is not None:
        model_args = model.get("model_args") or {}
        if model_args.get("lead_time_mode", False) and not dataset.get("provide_lead_time", False):
            problems.append("model lead_time_mode=True but dataset provide_lead_time=False")
        if model.get("prev_hr_conditioning", False) and not dataset.get("provide_prev_target", False):
            problems.append("model prev_hr_conditioning=True but dataset provide_prev_target=False")
    return problems


def test_entry_configs_are_found():
    assert len(ENTRY_CONFIGS) > 10, f"no entry configs found under {CONF_DIR}"


@pytest.mark.parametrize("name", ENTRY_CONFIGS)
def test_entry_config_describes_runnable_setup(name, request):
    if name in KNOWN_BROKEN:
        request.applymarker(pytest.mark.xfail(reason=KNOWN_BROKEN[name], strict=True))
    assert config_problems(compose_entry(name)) == []
