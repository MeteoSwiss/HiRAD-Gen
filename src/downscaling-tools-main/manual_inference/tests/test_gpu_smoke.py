from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
def test_cuda_tensor_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this node.")

    x = torch.ones(1024, device="cuda")
    y = torch.full((1024,), 2.0, device="cuda")
    z = x + y

    assert z.is_cuda
    assert torch.allclose(z, torch.full((1024,), 3.0, device="cuda"))


@pytest.mark.gpu
def test_cuda_autocast_fp16_matmul():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this node.")

    a = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        c = a @ b

    assert c.is_cuda
    assert c.dtype in (torch.float16, torch.float32)
    assert torch.isfinite(c).all()


@pytest.mark.gpu
def test_cuda_autocast_bf16_matmul():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this node.")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is not supported on this GPU.")

    a = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        c = a @ b

    assert c.is_cuda
    assert c.dtype in (torch.bfloat16, torch.float32)
    assert torch.isfinite(c).all()
