#!/bin/bash
#
# Minimal multi-node NCCL reproducer for:
#   cxil_map: write error
#   NET/OFI Unable to register memory (type = 1) for device 0. RC: -13, Error: Permission denied
#
# Observed in production on clariden/santis (CSCS Alps) when running a multi-node
# PyTorch DDP job inside a Pyxis/Enroot container with com.hooks.cxi.enabled = "false".
# Rank 12 (node 4 of 8) fails during the first cross-node NCCL ALLREDUCE inside
# DistributedDataParallel.__init__ and all other ranks time out 10 minutes later,
# causing nodes to get stuck in canceling state.
#
# The failure is inside DistributedDataParallel.__init__ at
# _verify_param_shape_across_processes (torch/nn/parallel/distributed.py:858).
# A plain torch.distributed allreduce succeeds (confirmed). This script tests
# whether DDP init itself — using the same flags as production — triggers the
# cxil_map memory registration failure.
#
# Expected failure (matching production):
#   cxil_map: write error
#   ncclSystemError: NET/OFI Unable to register memory ... Permission denied

#SBATCH --job-name="nccl-cxi-repro"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/nccl_cxi_repro.log
#SBATCH --error=./logs/nccl_cxi_repro.err

### ACCOUNT ###
#SBATCH -A c38

# Write the minimal Python test to the shared filesystem so all nodes can read it
REPRO_PY="${SLURM_SUBMIT_DIR}/logs/nccl_cxi_repro_${SLURM_JOB_ID}.py"
cat > "$REPRO_PY" << 'PYEOF'
import os
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

rank       = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ["SLURM_LOCALID"])
master     = os.environ["MASTER_ADDR"]
port       = os.environ["MASTER_PORT"]
hostname   = socket.gethostname()

print(f"[rank {rank:3d}] host={hostname}  local_rank={local_rank}  "
      f"world={world_size}  master={master}:{port}", flush=True)

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{master}:{port}",
    rank=rank,
    world_size=world_size,
)
print(f"[rank {rank:3d}] process group initialized", flush=True)

torch.cuda.set_device(local_rank)

# Production fails inside DistributedDataParallel.__init__ at
# _verify_param_shape_across_processes (distributed.py:858).
# Plain allreduce works fine. This tests whether DDP init itself
# triggers the cxil_map memory registration failure.
model = nn.Sequential(
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
).to(f"cuda:{local_rank}")

print(f"[rank {rank:3d}] wrapping model in DDP", flush=True)
ddp_model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    broadcast_buffers=True,
    output_device=local_rank,
    find_unused_parameters=True,
    bucket_cap_mb=35,
    gradient_as_bucket_view=True,
)
print(f"[rank {rank:3d}] DDP init done", flush=True)

# One forward + backward to exercise the gradient buckets
x = torch.randn(4, 256, device=f"cuda:{local_rank}")
loss = ddp_model(x).sum()
loss.backward()
print(f"[rank {rank:3d}] forward/backward done", flush=True)

dist.destroy_process_group()
print(f"[rank {rank:3d}] clean exit", flush=True)
PYEOF

# Same environment setup as production job
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

# Extra NCCL diagnostics to help support pinpoint the failure
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python $REPRO_PY
"
