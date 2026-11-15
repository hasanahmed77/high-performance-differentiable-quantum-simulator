"""Test-suite configuration."""

import importlib.util
from typing import Any

import pytest

from ketjet.backends import Backend, get_backend

# PennyLane is an optional extra (ketjet[pennylane]) and part of the test group. Tests that need it
# use pytest.importorskip, so the suite still runs without it; the shared helper modules must not
# even be collected (pytest imports every module for --doctest-modules) when it's missing.
collect_ignore = (
    []
    if importlib.util.find_spec("pennylane")
    else ["pennylane_support.py", "regression/generate_golden.py"]
)
if not importlib.util.find_spec("jax"):
    collect_ignore += ["unit/test_jax_backend.py", "unit/test_compile_circuit.py"]


@pytest.fixture(params=["numpy", "jax"])
def backend(request: pytest.FixtureRequest) -> Backend[Any]:
    """Every backend in turn (ADR-0001 D4); the JAX one is skipped when JAX isn't installed."""
    if request.param == "jax":
        pytest.importorskip("jax")
    return get_backend(request.param)


@pytest.fixture(params=["numpy", "jax"])
def backend_name(request: pytest.FixtureRequest) -> str:
    """Every backend's name in turn, for APIs that take one (``KetjetQubit(backend=...)``)."""
    if request.param == "jax":
        pytest.importorskip("jax")
    return str(request.param)
