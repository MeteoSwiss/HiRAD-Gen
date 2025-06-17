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


"""Miscellaneous utility classes and functions."""

import datetime
from typing import Iterator

import cftime
import numpy as np
import torch

# ruff: noqa: E722 PERF203 S110 E713 S324


class StackedRandomGenerator:  # pragma: no cover
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


# Small util functions
# -------------------------------------------------------------------------------------

def time_range(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step: datetime.timedelta,
    inclusive: bool = False,
):
    """Like the Python `range` iterator, but with datetimes."""
    t = start_time
    while (t <= end_time) if inclusive else (t < end_time):
        yield t
        t += step

def get_time_from_range(times_range, time_format="%Y-%m-%dT%H:%M:%S"):
    """Generates a list of times within a given range.

    Args:
        times_range: A list containing start time, end time, and optional interval (hours).
        time_format: The format of the input times (default: "%Y-%m-%dT%H:%M:%S").

    Returns:
        A list of times within the specified range.
    """

    start_time = datetime.datetime.strptime(times_range[0], time_format)
    end_time = datetime.datetime.strptime(times_range[1], time_format)
    interval = (
        datetime.timedelta(hours=times_range[2])
        if len(times_range) > 2
        else datetime.timedelta(hours=1)
    )

    times = [
        t.strftime(time_format)
        for t in time_range(start_time, end_time, interval, inclusive=True)
    ]
    return times


# ----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


class InfiniteSampler(torch.utils.data.Sampler[int]):  # pragma: no cover
    """Sampler for torch.utils.data.DataLoader that loops over the dataset indefinitely.

    This sampler yields indices indefinitely, optionally shuffling items as it goes.
    It can also perform distributed sampling when rank and num_replicas are specified.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from
    rank : int, default=0
        The rank of the current process within num_replicas processes
    num_replicas : int, default=1
        The number of processes participating in distributed sampling
    shuffle : bool, default=True
        Whether to shuffle the indices
    seed : int, default=0
        Random seed for reproducibility when shuffling
    window_size : float, default=0.5
        Fraction of dataset to use as window for shuffling. Must be between 0 and 1.
        A larger window means more thorough shuffling but slower iteration.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        window_size: float = 0.5,
    ):
        if not len(dataset) > 0:
            raise ValueError("Dataset must contain at least one item")
        if not num_replicas > 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be non-negative and less than num_replicas")
        if not 0 <= window_size <= 1:
            raise ValueError("window_size must be between 0 and 1")
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self) -> Iterator[int]:
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
