"""Shared pytest fixtures. Architect-owned (protected)."""

import os

import pytest


@pytest.fixture(scope="session")
def npu():
    """Fail (never skip) when the NPU stack is not usable.

    Tests that run on hardware take this fixture. A missing device must fail the
    done-command instead of silently passing it.
    """
    if not os.path.exists("/dev/accel/accel0"):
        pytest.fail("/dev/accel/accel0 missing: amdxdna driver not loaded?")
    try:
        import pyxrt  # noqa: F401
    except ImportError as e:
        pytest.fail(f"pyxrt not importable ({e}); run tests via scripts/test.sh")
    return "npu2"


@pytest.fixture(scope="session")
def igpu():
    """Fail (never skip) when the iGPU path (llama.cpp shim + Vulkan) is not usable."""
    from tests.reference.models import SHIM

    if not SHIM.exists():
        pytest.fail(f"{SHIM} missing: run scripts/build-llama-shim.sh")
    if not os.path.exists("/dev/dri/renderD128"):
        pytest.fail("/dev/dri/renderD128 missing: no GPU render node")
    return "vulkan"
