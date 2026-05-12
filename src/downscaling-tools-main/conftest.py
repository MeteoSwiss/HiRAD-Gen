from __future__ import annotations

import os
import signal
import time
from typing import Iterator

import pytest


TEST_TIMEOUT_SECONDS = 30 * 60


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.gpu.",
    )
    parser.addoption(
        "--gpu-debug",
        action="store_true",
        default=False,
        help="Enable CUDA debug-oriented environment variables for GPU tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: test requires a CUDA-capable GPU")

    if config.getoption("--gpu-debug"):
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="GPU test: pass --run-gpu to execute.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _enforce_30min_max_per_test() -> Iterator[None]:
    def _handle_timeout(signum: int, frame: object) -> None:
        raise TimeoutError(f"Test exceeded {TEST_TIMEOUT_SECONDS} seconds.")

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(TEST_TIMEOUT_SECONDS)
    start = time.monotonic()

    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        if elapsed > TEST_TIMEOUT_SECONDS:
            pytest.fail(
                f"Test exceeded {TEST_TIMEOUT_SECONDS} seconds ({elapsed:.1f}s).",
                pytrace=False,
            )
