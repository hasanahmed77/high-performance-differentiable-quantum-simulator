"""Unit tests for the backend protocol, run against every backend (QS-209, QS-602).

The ``backend`` fixture yields each backend in turn. Tests use only what the protocol promises:
states are opaque, and amplitudes are read through ``to_numpy``.
"""

from typing import Any

import numpy as np
import pytest

from ketjet.backends import Backend, NumPyBackend, get_backend
from ketjet.observables import X as PauliX
from ketjet.observables import Z as PauliZ
from ketjet.operations import CNOT, CRZ, RX, RY, H, Operation, Parameter, Toffoli, X
from ketjet.statevector import StateVector
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]


def test_numpy_backend_satisfies_the_protocol_statically() -> None:
    backend: Backend[StateVector] = NumPyBackend()  # checked by mypy --strict
    assert backend.name == "numpy"


def test_every_backend_satisfies_the_protocol_at_runtime(backend: Backend[Any]) -> None:
    assert isinstance(backend, Backend)
    assert get_backend(backend.name).name == backend.name


def test_objects_missing_methods_do_not_satisfy_the_protocol() -> None:
    class Incomplete:
        name = "incomplete"

        def zero_state(self, n_wires: int, dtype: Any = None) -> None: ...

    assert not isinstance(Incomplete(), Backend)


def test_get_backend_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match=r"Unknown backend 'cuda'. Available: jax, numpy."):
        get_backend("cuda")


def test_get_backend_explains_how_to_install_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry in sys.modules makes the import fail, as it does without JAX installed.
    monkeypatch.setitem(__import__("sys").modules, "ketjet.backends.jax_backend", None)
    with pytest.raises(ImportError, match=r'pip install "ketjet\[jax\]"'):
        get_backend("jax")


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_state_and_n_wires(backend: Backend[Any], dtype: Any) -> None:
    state = backend.zero_state(3, dtype)
    assert backend.n_wires(state) == 3
    assert state.dtype == np.dtype(dtype)
    expected = np.zeros(8)
    expected[0] = 1
    np.testing.assert_array_equal(backend.to_numpy(state), expected)


@pytest.mark.parametrize("n_wires", [0, 64, 1.5, True])
def test_zero_state_rejects_invalid_qubit_counts(backend: Backend[Any], n_wires: Any) -> None:
    with pytest.raises((ValueError, TypeError), match="n_wires must be"):
        backend.zero_state(n_wires)


def test_zero_state_rejects_unsupported_precision(backend: Backend[Any]) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        backend.zero_state(2, "float64")


def test_apply_operation_is_pure(backend: Backend[Any]) -> None:
    state = backend.zero_state(2)
    before = backend.to_numpy(state)
    out = backend.apply_operation(state, H(0))
    np.testing.assert_array_equal(backend.to_numpy(state), before)
    assert out is not state


def test_to_numpy_returns_an_independent_writable_copy(backend: Backend[Any]) -> None:
    state = backend.zero_state(1)
    exported = backend.to_numpy(state)
    assert isinstance(exported, np.ndarray)
    exported[0] = 0
    assert backend.to_numpy(state)[0] == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_controlled_operations_route_controls_correctly(backend: Backend[Any], dtype: Any) -> None:
    # |110> --Toffoli(0,1,2)--> |111>; with the control order swapped it's the same gate.
    state = backend.apply_operations(backend.zero_state(3, dtype), [X(0), X(1), Toffoli(0, 1, 2)])
    expected = np.zeros(8)
    expected[7] = 1
    assert_close(backend.to_numpy(state), expected, dtype)
    assert state.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "op",
    [
        Operation("RX", (0,), (0.7,), controls=(2,), control_values=(False,)),
        Operation("RY", (2,), (-1.1,), controls=(0, 1), control_values=(True, False)),
        Operation("RZ", (1,), (0.4,), controls=(2,)),  # diagonal, control after the target
        Operation("ISWAP", (2, 0), (), controls=(1,), control_values=(False,)),
        Operation("PhaseShift", (0,), (2.2,), controls=(1, 2), control_values=(False, False)),
        CRZ(0.9, 2, 0),
    ],
    ids=repr,
)
def test_control_values_and_wire_order_match_the_full_matrix(
    backend: Backend[Any], op: Operation, dtype: Any
) -> None:
    """The backend's control handling equals the explicit controlled matrix on a generic state."""
    prepare = [RY(0.3 + 0.5 * w, w) for w in range(3)] + [RX(1.3 - 0.4 * w, w) for w in range(3)]
    psi = backend.apply_operations(backend.zero_state(3, dtype), prepare)
    host = backend.to_numpy(psi)
    assert len(set(np.round(np.abs(host), 3))) == 8  # asymmetric: no two amplitudes alike

    # The full matrix on (controls + targets), embedded by hand: an independent reference.
    wires = op.all_wires
    reference = np.zeros(8, dtype=np.complex128)
    matrix = op.matrix()
    for column in range(8):
        bits = [(column >> (2 - w)) & 1 for w in range(3)]
        sub_in = int("".join(str(bits[w]) for w in wires), 2)
        for sub_out in range(1 << len(wires)):
            out_bits = bits.copy()
            for position, w in enumerate(wires):
                out_bits[w] = (sub_out >> (len(wires) - 1 - position)) & 1
            row = int("".join(map(str, out_bits)), 2)
            reference[row] += matrix[sub_out, sub_in] * host[column]
    assert_close(backend.to_numpy(backend.apply_operation(psi, op)), reference, dtype)


def test_apply_operations_on_empty_sequence_returns_state_unchanged(backend: Backend[Any]) -> None:
    state = backend.zero_state(2)
    out = backend.to_numpy(backend.apply_operations(state, []))
    np.testing.assert_array_equal(out, backend.to_numpy(state))


def test_wire_outside_the_state_is_rejected_with_context(backend: Backend[Any]) -> None:
    with pytest.raises(ValueError, match=r"X uses wire\(s\) \[2\], but the state has 2 qubit"):
        backend.apply_operation(backend.zero_state(2), CNOT(0, 2))


def test_unbound_parameters_are_rejected(backend: Backend[Any]) -> None:
    with pytest.raises(ValueError, match="unbound parameter"):
        backend.apply_operation(backend.zero_state(1), RX(Parameter("t"), 0))


# --------------------------------------------------------------------------- measurements


@pytest.mark.parametrize("dtype", DTYPES)
def test_measurements_return_numpy_types_in_the_states_precision(
    backend: Backend[Any], dtype: Any
) -> None:
    state = backend.apply_operations(backend.zero_state(2, dtype), [RY(0.6, 0), CNOT(0, 1)])
    p = backend.probs(state, [1])
    assert isinstance(p, np.ndarray)
    assert p.dtype == (np.float64 if dtype is np.complex128 else np.float32)
    assert p.flags.writeable
    assert_close(p, [np.cos(0.3) ** 2, np.sin(0.3) ** 2], dtype)
    assert type(backend.expval(state, PauliZ(0))) is float
    assert type(backend.var(state, PauliZ(0))) is float
    assert_close(backend.expval(state, PauliZ(0)), np.cos(0.6), dtype)
    assert_close(backend.var(state, PauliZ(0)), np.sin(0.6) ** 2, dtype)
    bits = backend.sample(state, 50, seed=3)
    assert isinstance(bits, np.ndarray)
    assert bits.dtype == np.int64
    assert bits.shape == (50, 2)
    assert bool((bits[:, 0] == bits[:, 1]).all())


def test_measurements_validate_their_arguments(backend: Backend[Any]) -> None:
    state = backend.zero_state(2)
    with pytest.raises(ValueError, match="Wire 2 is out of range for a 2-qubit state"):
        backend.probs(state, [2])
    with pytest.raises(ValueError, match="Wires must be distinct"):
        backend.probs(state, [0, 0])
    with pytest.raises(ValueError, match=r"acts on wire\(s\) \[3\], but the state has 2 qubit"):
        backend.expval(state, PauliX(3))
    with pytest.raises(ValueError, match=r"acts on wire\(s\) \[3\], but the state has 2 qubit"):
        backend.var(state, PauliX(3))
    with pytest.raises(ValueError, match="shots must be a positive int"):
        backend.sample(state, 0)
    with pytest.raises(ValueError, match="Wire 5 is out of range"):
        backend.sample(state, 10, [5])


def test_variance_of_an_eigenstate_is_zero_and_never_negative(backend: Backend[Any]) -> None:
    for angle in np.linspace(0.1, 3.0, 12):  # RY(θ)|0⟩ is an eigenstate of cos(θ) Z + sin(θ) X
        state = backend.apply_operation(backend.zero_state(1), RY(float(angle), 0))
        observable = float(np.cos(angle)) * PauliZ(0) + float(np.sin(angle)) * PauliX(0)
        assert 0.0 <= backend.var(state, observable) < 1e-12


# --------------------------------------------------------------------------- adjoint Jacobian


def test_adjoint_jacobian_of_a_rotation(backend: Backend[Any]) -> None:
    jac = backend.adjoint_jacobian(1, [RX(0.5, 0)], [PauliZ(0)], [(0, 0)])
    assert jac.dtype == np.float64
    assert jac.shape == (1, 1)
    assert_close(jac, [[-np.sin(0.5)]], np.complex128)


def test_adjoint_jacobian_edge_cases(backend: Backend[Any]) -> None:
    ops = [RX(0.5, 0), CNOT(0, 1)]
    assert backend.adjoint_jacobian(2, ops, [PauliZ(0)], []).shape == (1, 0)
    assert backend.adjoint_jacobian(2, ops, [], [(0, 0)]).shape == (0, 1)
    # Negative indices must not wrap around to a real slot.
    for slot in [(2, 0), (0, 1), (1, 0), (-1, 0), (-2, 0), (0, -1)]:
        with pytest.raises(ValueError, match=r"Trainable slot .* doesn't exist in the circuit"):
            backend.adjoint_jacobian(2, ops, [PauliZ(0)], [slot])
    with pytest.raises(ValueError, match="unbound parameter"):
        backend.adjoint_jacobian(1, [RX(Parameter("t"), 0)], [PauliZ(0)], [(0, 0)])
