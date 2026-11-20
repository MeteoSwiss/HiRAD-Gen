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

import atexit
import os
from warnings import warn

import numpy as np
import torch
import torch.distributed as dist


class UninitializedDistributedManagerWarning(Warning):
    """Warning to indicate usage of an uninitialized DistributedManager"""

    def __init__(self):
        message = (
            "A DistributedManager object is being instantiated before "
            + "this singleton class has been initialized. Instantiating a manager before "
            + "initialization can lead to unexpected results where processes fail "
            + "to communicate. Initialize the distributed manager via "
            + "DistributedManager.initialize() before instantiating."
        )
        super().__init__(message)


class DistributedManager(object):
    """Distributed Manager for setting up distributed training environment.

    This is a singleton that creates a persistance class instance for storing parallel
    environment information through out the life time of the program. This should be
    used to help set up Distributed Data Parallel and parallel datapipes.

    Note
    ----
    One should call `DistributedManager.initialize()` prior to constructing a manager
    object

    Example
    -------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """

    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not hasattr(obj, "_cuda"):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, "_broadcast_buffers"):
            obj._broadcast_buffers = False
        if not hasattr(obj, "_find_unused_parameters"):
            obj._find_unused_parameters = False
        if not hasattr(obj, "_initialization_method"):
            obj._initialization_method = "None"
        if not hasattr(obj, "_is_initialized"):
            obj._is_initialized = False

        return obj

    def __init__(self):
        if not self._is_initialized:
            raise UninitializedDistributedManagerWarning()
        super().__init__()

    @property
    def rank(self):
        """Process rank"""
        return self._rank

    @property
    def local_rank(self):
        """Process rank on local machine"""
        return self._local_rank

    @property
    def world_size(self):
        """Number of processes in distributed environment"""
        return self._world_size

    @property
    def device(self):
        """Process device"""
        return self._device

    @property
    def distributed(self):
        """Distributed environment"""
        return self._distributed

    @property
    def cuda(self):
        """If cuda is available"""
        return self._cuda

    @property
    def broadcast_buffers(self):
        """broadcast_buffers in PyTorch DDP"""
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: bool):
        """Setter for broadcast_buffers"""
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        """find_unused_parameters in PyTorch DDP"""
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: bool):
        """Setter for find_unused_parameters"""
        if find_params:
            warn(
                "Setting `find_unused_parameters` in DDP to true, "
                "use only if necessary."
            )
        self._find_unused_parameters = find_params

    def __str__(self):
        output = (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self._initialization_method}'. Device set to {str(self.device)}"
        )
        return output

    @classmethod
    def is_initialized(cls) -> bool:
        """If manager singleton has been initialized"""
        return cls._shared_state.get("_is_initialized", False)

    @staticmethod
    def get_available_backend():
        """Get communication backend"""
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        """Setup method using generic initialization"""
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        if "LOCAL_RANK" in os.environ:
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                local_rank = int(local_rank)
            else:
                local_rank = rank % torch.cuda.device_count()

        else:
            local_rank = rank % torch.cuda.device_count()

        # Read env variables
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        """Setup method using OpenMPI initialization"""
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        """Setup method using SLURM initialization"""
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NPROCS"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        addr = os.environ.get("MASTER_ADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        """
        Initialize distributed manager

        Current supported initialization methods are:
            `ENV`: PyTorch environment variable initialization
                https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            `SLURM`: Initialization on SLURM systems.
                Uses `SLURM_PROCID`, `SLURM_NPROCS`, `SLURM_LOCALID` and
                `SLURM_LAUNCH_NODE_IPADDR` environment variables.
            `OPENMPI`: Initialization for OpenMPI launchers.
                Uses `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE` and
                `OMPI_COMM_WORLD_LOCAL_RANK` environment variables.

        Initialization by default is done using the first valid method in the order
        listed above. Initialization method can also be explicitly controlled using the
        `DISTRIBUTED_INITIALIZATION_METHOD` environment variable and setting it
        to one of the options above.
        """
        if DistributedManager.is_initialized():
            warn("Distributed manager is already intialized")
            return

        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        # https://pytorch.org/docs/master/notes/cuda.html#id5
        # was changed in version 2.2
        #TODO why is setting this important?
        if torch.__version__ < (2, 2):
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        else:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        initialization_method = os.getenv(
            "DISTRIBUTED_INITIALIZATION_METHOD"
        )
        if initialization_method is None:
            try:
                DistributedManager.initialize_env()
            except TypeError:
                if "SLURM_PROCID" in os.environ:
                    DistributedManager.initialize_slurm(port)
                elif "OMPI_COMM_WORLD_RANK" in os.environ:
                    DistributedManager.initialize_open_mpi(addr, port)
                else:
                    warn(
                        "Could not initialize using ENV, SLURM or OPENMPI methods. Assuming this is a single process job"
                    )
                    DistributedManager._shared_state["_is_initialized"] = True
        elif initialization_method == "ENV":
            DistributedManager.initialize_env()
        elif initialization_method == "SLURM":
            DistributedManager.initialize_slurm(port)
        elif initialization_method == "OPENMPI":
            DistributedManager.initialize_open_mpi(addr, port)
        else:
            raise RuntimeError(
                "Unknown initialization method "
                f"{initialization_method}. "
                "Supported values for "
                "DISTRIBUTED_INITIALIZATION_METHOD are "
                "ENV, SLURM and OPENMPI"
            )

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        """Set up PyTorch distributed process group and update manager attributes"""
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        DistributedManager._shared_state["_is_initialized"] = True
        manager = DistributedManager()

        manager._distributed = torch.distributed.is_available() and world_size > 1
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank

        manager._device = torch.device(
            f"cuda:{manager.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        #TODO device_id makes the init hang, couldn't figure out why
        if manager._distributed:
            # Setup distributed process group
            try:
                dist.init_process_group(
                    backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    device_id=manager.device, 
                )
            #           rank=manager.rank,
            #           world_size=manager.world_size,
            #           device_id=manager.device,
            except TypeError:
                # device_id only introduced in PyTorch 2.3
                dist.init_process_group(
                    backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                )

        if torch.cuda.is_available():
            # Set device for this process and empty cache to optimize memory usage
            torch.cuda.set_device(manager.device)
            torch.cuda.device(manager.device)
            torch.cuda.empty_cache()

        manager._initialization_method = method

    @atexit.register
    @staticmethod
    def cleanup():
        """Clean up distributed group and singleton"""
        # Destroying group.WORLD is enough for all process groups to get destroyed
        if (
            "_is_initialized" in DistributedManager._shared_state
            and DistributedManager._shared_state["_is_initialized"]
            and "_distributed" in DistributedManager._shared_state
            and DistributedManager._shared_state["_distributed"]
        ):
            if torch.cuda.is_available():
                dist.barrier(device_ids=[DistributedManager().local_rank])
            else:
                dist.barrier()
            dist.destroy_process_group()
        DistributedManager._shared_state = {}
