"""Unit tests for the backend protocol and the NumPy reference backend (QS-209)."""

from typing import Any

import numpy as np
import pytest

from ketjet.backends import Backend, NumPyBackend
from ketjet.operations import CNOT, RX, H, Parameter, Toffoli, X
from ketjet.statevector import StateVector
from tests.tolerances import assert_close


def test_numpy_backend_satisfies_the_protocol_statically_and_at_runtime() -> None:
    backend: Backend[StateVector] = NumPyBackend()  # checked by mypy --strict
    assert isinstance(backend, Backend)
    assert backend.name == "numpy"


def test_objects_missing_methods_do_not_satisfy_the_protocol() -> None:
    class Incomplete:
        name = "incomplete"

        def zero_state(self, n_wires: int, dtype: Any = None) -> None: ...

    assert not isinstance(Incomplete(), Backend)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_zero_state_and_n_wires(dtype: Any) -> None:
    backend = NumPyBackend()
    state = backend.zero_state(3, dtype)
    assert backend.n_wires(state) == 3
    assert state.dtype == np.dtype(dtype)


def test_apply_operation_is_pure() -> None:
    backend = NumPyBackend()
    state = backend.zero_state(2)
    before = state.copy()
    out = backend.apply_operation(state, H(0))
    np.testing.assert_array_equal(state, before)
    assert out is not state


def test_to_numpy_returns_an_independent_copy() -> None:
    backend = NumPyBackend()
    state = backend.zero_state(1)
    exported = backend.to_numpy(state)
    exported[0] = 0
    assert state[0] == 1


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_controlled_operations_route_controls_correctly(dtype: Any) -> None:
    backend = NumPyBackend()
    # |110> --Toffoli(0,1,2)--> |111>; with the control order swapped it's the same gate.
    state = backend.apply_operations(backend.zero_state(3, dtype), [X(0), X(1), Toffoli(0, 1, 2)])
    expected = np.zeros(8)
    expected[7] = 1
    assert_close(backend.to_numpy(state), expected, dtype)
    assert state.dtype == np.dtype(dtype)


def test_apply_operations_on_empty_sequence_returns_state_unchanged() -> None:
    backend = NumPyBackend()
    state = backend.zero_state(2)
    np.testing.assert_array_equal(backend.apply_operations(state, []), state)


def test_wire_outside_the_state_is_rejected_with_context() -> None:
    backend = NumPyBackend()
    with pytest.raises(ValueError, match=r"X uses wire\(s\) \[2\], but the state has 2 qubit"):
        backend.apply_operation(backend.zero_state(2), CNOT(0, 2))


def test_unbound_parameters_are_rejected() -> None:
    backend = NumPyBackend()
    with pytest.raises(ValueError, match="unbound parameter"):
        backend.apply_operation(backend.zero_state(1), RX(Parameter("t"), 0))
