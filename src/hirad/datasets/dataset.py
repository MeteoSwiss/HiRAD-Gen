# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Tuple, Union
import copy
import torch

from hirad.utils.function_utils import InfiniteSampler
from hirad.distributed import DistributedManager

from .era5_cosmo import ERA5_COSMO
from .era5_real import ERA5_REAL
from .anemoi_dataset import ANEMOI_ERA5_COSMO, ANEMOI_ERA5_REAL
from .base import DownscalingDataset


# this maps all known dataset types to the corresponding init function
known_datasets = {
    "era5_cosmo": ERA5_COSMO,
    "era5_real": ERA5_REAL,
    "anemoi_era5_cosmo": ANEMOI_ERA5_COSMO,
    "anemoi_era5_real": ANEMOI_ERA5_REAL,
}


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    train_test_split: bool = True,
    sampler_start_idx: int = 0,
) -> Tuple[
    DownscalingDataset,
    Iterable,
    Union[DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    A wrapper function for managing the train-test split for the CWB dataset.

    Parameters:
    - dataset_cfg (dict): Configuration for the dataset.
    - dataloader_cfg (dict, optional): Configuration for the dataloader. Defaults to None.
    - batch_size (int): The number of samples in each batch of data. Defaults to 1.
    - seed (int): The random seed for dataset shuffling. Defaults to 0.
    - train_test_split (bool): A flag to determine whether to create a validation dataset. Defaults to True.
    - sampler_start_idx (int): The initial index of the sampler to use for resuming training. Defaults to 0.

    Returns:
    - Tuple[base.DownscalingDataset, Iterable, Optional[base.DownscalingDataset], Optional[Iterable]]: A tuple containing the training dataset and iterator, and optionally the validation dataset and iterator if train_test_split is True.
    """

    config = copy.deepcopy(dataset_cfg)
    config.pop("validation", None)
    config.pop("validation_start_date", None)
    config.pop("validation_end_date", None)
    (dataset, dataset_iter) = init_dataset_from_config(
        config, dataloader_cfg, batch_size=batch_size, seed=seed, sampler_start_idx=sampler_start_idx,
    )
    if train_test_split:
        valid_dataset_cfg = copy.deepcopy(dataset_cfg)
        del valid_dataset_cfg['validation']
        if "validation_start_date" not in valid_dataset_cfg or "validation_end_date" not in valid_dataset_cfg:
            raise ValueError("validation_start_date and validation_en_date must be specified in anemoi dataset_cfg when validation is set to True")
        valid_dataset_cfg["start_date"] = valid_dataset_cfg["validation_start_date"]
        valid_dataset_cfg["end_date"] = valid_dataset_cfg["validation_end_date"]
        del valid_dataset_cfg['validation_start_date']
        del valid_dataset_cfg['validation_end_date']

        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    sampler_start_idx: int = 0,
    pop_type: bool = True,
) -> Tuple[DownscalingDataset, Iterable]:

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.get("type", "era5_cosmo")
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dist = DistributedManager()
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed, start_idx=sampler_start_idx,
    )

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)


def get_dataset_and_sampler_inference(dataset_cfg, times, has_lead_time=False):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    # if has_lead_time:
    #     plot_times = times
    # else:
    #     plot_times = [
    #         datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
    #         for time in times
    #     ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in times]
    sampler = time_indices

    return dataset, sampler
