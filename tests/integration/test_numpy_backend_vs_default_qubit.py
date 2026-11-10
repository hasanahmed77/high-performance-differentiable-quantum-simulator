"""End-to-end: circuits run through the NumPy backend equal PennyLane's default.qubit (QS-209).

The PennyLane circuit is built from PennyLane's own native gates (not from our matrices), so this
checks gate definitions, control handling, wire order and the kernel together against an
independent simulator.
"""

from typing import Any

import numpy as np
import pytest

from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from tests.tolerances import assert_close

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

from tests.pennylane_support import random_circuit, run_default_qubit  # noqa: E402


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_bell_state_matches_default_qubit(dtype: Any) -> None:
    circuit = [ops.H(0), ops.CNOT(0, 1)]
    backend = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(2, dtype), circuit)
    assert_close(backend.to_numpy(state), run_default_qubit(circuit, 2, qml.state), dtype)
    assert_close(backend.to_numpy(state), np.array([1, 0, 0, 1]) / np.sqrt(2), dtype)


@pytest.mark.parametrize("n_wires", [3, 5, 8])
def test_ghz_state_matches_default_qubit(n_wires: int) -> None:
    circuit = [ops.H(0)] + [ops.CNOT(i, i + 1) for i in range(n_wires - 1)]
    backend = NumPyBackend()
    state = backend.to_numpy(backend.apply_operations(backend.zero_state(n_wires), circuit))
    expected = np.zeros(2**n_wires)
    expected[0] = expected[-1] = 1 / np.sqrt(2)
    assert_close(state, expected, np.complex128)
    assert_close(state, run_default_qubit(circuit, n_wires, qml.state), np.complex128)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("seed", range(25))
def test_random_circuits_match_default_qubit(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 7))
    circuit = random_circuit(n_wires, depth=30, rng=rng)
    backend = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(n_wires, dtype), circuit)
    assert state.dtype == np.dtype(dtype)
    assert_close(backend.to_numpy(state), run_default_qubit(circuit, n_wires, qml.state), dtype)
