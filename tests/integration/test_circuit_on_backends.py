"""The ``Circuit`` API gives the same answers on every backend (QS-602).

Users pick a backend with one argument; nothing else about their program changes.
"""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit, gradients
from ketjet.backends import Backend
from ketjet.observables import Hermitian, X, Y, Z
from ketjet.operations import Parameter
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
A, B, C = Parameter("a"), Parameter("b"), Parameter("c")
VALUES = {"a": 0.37, "b": -1.21, "c": 2.05}
HAMILTONIAN = (
    0.6 * (Z(0) @ Z(1))
    - 0.9 * X(2)
    + 0.25 * (Y(0) @ X(1) @ Z(2))
    + Hermitian([[1, 1j], [-1j, 0]], [1])
)


def _ansatz() -> Circuit:
    return (
        Circuit(3)
        .ry(A, 0)
        .rx(B, 1)
        .cnot(0, 1)
        .crz(C, 1, 2)
        .rot(A, 0.4, B, 2)  # shared symbols
        .toffoli(0, 2, 1)
        .u3(C, 0.3, A, 0)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_measurements_match_the_default_backend(backend: Backend[Any], dtype: Any) -> None:
    circuit = _ansatz().bind(VALUES)
    state = circuit.state(backend=backend, dtype=dtype)
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.dtype(dtype)
    assert_close(state, circuit.state(dtype=dtype), dtype)
    assert_close(
        circuit.probs([2, 0], backend=backend, dtype=dtype),
        circuit.probs([2, 0], dtype=dtype),
        dtype,
    )
    assert_close(
        circuit.expval(HAMILTONIAN, backend=backend, dtype=dtype),
        circuit.expval(HAMILTONIAN, dtype=dtype),
        dtype,
    )
    assert_close(
        circuit.var(HAMILTONIAN, backend=backend, dtype=dtype),
        circuit.var(HAMILTONIAN, dtype=dtype),
        dtype,
    )


def test_seeded_samples_and_shot_estimates_match_the_default_backend(
    backend: Backend[Any],
) -> None:
    circuit = _ansatz().bind(VALUES)
    np.testing.assert_array_equal(
        circuit.sample(300, [1, 2], seed=5, backend=backend), circuit.sample(300, [1, 2], seed=5)
    )
    estimate = circuit.expval(HAMILTONIAN, shots=2000, seed=9, backend=backend)
    assert estimate == pytest.approx(circuit.expval(HAMILTONIAN, shots=2000, seed=9), abs=1e-12)


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift", "finite-diff"])
def test_gradients_match_the_default_backend(backend: Backend[Any], method: Any) -> None:
    observables = [HAMILTONIAN, Z(1)]
    jac = gradients.jacobian(_ansatz(), observables, VALUES, method=method, backend=backend)
    reference = gradients.jacobian(_ansatz(), observables, VALUES, method="adjoint")
    assert jac.shape == (2, 3)
    assert np.abs(reference[0]).min() > 1e-3  # every Hamiltonian entry is a real derivative
    if method == "finite-diff":
        np.testing.assert_allclose(jac, reference, atol=1e-5)
    else:
        assert_close(jac, reference, np.complex128)
    gradient = _ansatz().gradient(HAMILTONIAN, VALUES, method=method, backend=backend)
    np.testing.assert_allclose(gradient, reference[0], atol=1e-5)
