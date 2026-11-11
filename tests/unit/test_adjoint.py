"""Unit tests for adjoint differentiation (QS-404, ADR-0004 D3, D4)."""

import tracemalloc
from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet import operations as ops
from ketjet.adjoint import adjoint_jacobian
from ketjet.backends import Backend, NumPyBackend
from ketjet.gates import GATES, GateSpec
from ketjet.gradients import grad, jacobian
from ketjet.observables import Hermitian, Observable, X, Y, Z
from ketjet.operations import Operation, Parameter
from ketjet.statevector import state_nbytes
from tests.random_circuits import random_circuit, symbolize
from tests.tolerances import GRADIENT_FD, assert_close

a, b, t = Parameter("a"), Parameter("b"), Parameter("t")
PARAMETRIZED = sorted(name for name, spec in GATES.items() if spec.num_params)


def _observables(n_wires: int, rng: np.random.Generator) -> list[Observable]:
    """One observable of every type: Pauli, Sum, two-wire Hermitian, Prod with a Hermitian."""
    m2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    m1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    w = [int(x) for x in rng.permutation(n_wires)]
    return [
        Z(w[0]),
        0.4 * Z(w[0]) @ Z(w[1]) - 1.3 * X(w[1]) + 0.2 * Y(w[0]),
        Hermitian(m2 + m2.conj().T, wires=(w[0], w[1])),
        Hermitian(m1 + m1.conj().T, wires=(w[1],)) @ X(w[0]),
    ]


# --------------------------------------------------------------------------- derivatives


@pytest.mark.parametrize("name", PARAMETRIZED)
def test_derivative_equals_central_differences_of_the_matrix(name: str) -> None:
    spec = GATES[name]
    rng = np.random.default_rng(len(name))
    h = 1e-6
    for _ in range(20):
        params = tuple(float(x) for x in rng.uniform(-np.pi, np.pi, spec.num_params))
        for j in range(spec.num_params):
            up = tuple(p + h if k == j else p for k, p in enumerate(params))
            down = tuple(p - h if k == j else p for k, p in enumerate(params))
            fd = (spec.matrix(up) - spec.matrix(down)) / (2 * h)
            np.testing.assert_allclose(spec.derivative(params, j), fd, atol=GRADIENT_FD.atol)


def test_derivative_has_the_requested_precision() -> None:
    assert GATES["RX"].derivative((0.3,), 0, np.complex64).dtype == np.complex64


def test_derivative_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="takes 1 parameter"):
        GATES["RX"].derivative((0.1, 0.2), 0)
    with pytest.raises(ValueError, match="has no parameter 1"):
        GATES["RX"].derivative((0.1,), 1)
    with pytest.raises(ValueError, match="has no parameter 0"):
        GATES["H"].derivative((), 0)


def test_parametrized_gate_spec_requires_a_derivative() -> None:
    rx = GATES["RX"]
    with pytest.raises(ValueError, match="has no derivative_fn"):
        GateSpec("Bad", 1, 1, rx.matrix_fn, rx.inverse_fn, generator_eigvals=((-0.5, 0.5),))


@pytest.mark.parametrize("values", [(True,), (False,), (True, False)])
def test_controlled_derivative_is_zero_off_the_active_block(values: tuple[bool, ...]) -> None:
    controls = tuple(range(len(values)))
    op = Operation("RY", (len(values),), (0.7,), controls, values)
    d = op.derivative_matrix(0)
    trigger = int("".join("1" if v else "0" for v in values), 2)
    block = slice(2 * trigger, 2 * trigger + 2)
    expected = np.zeros_like(d)
    expected[block, block] = GATES["RY"].derivative((0.7,), 0)
    np.testing.assert_array_equal(d, expected)
    # It's the derivative of the full controlled matrix: identity blocks are constant.
    h = 1e-6
    up = Operation("RY", op.wires, (0.7 + h,), controls, values).matrix()
    down = Operation("RY", op.wires, (0.7 - h,), controls, values).matrix()
    np.testing.assert_allclose(d, (up - down) / (2 * h), atol=GRADIENT_FD.atol)


# --------------------------------------------------------------------------- agreement


def test_default_method_is_adjoint() -> None:
    c = Circuit(2).ry(a, 0).crx(b, 0, 1)
    values = {"a": 0.3, "b": 1.2}
    expected = grad(c, Z(1), values, method="adjoint")
    np.testing.assert_array_equal(grad(c, Z(1), values), expected)
    np.testing.assert_array_equal(c.gradient(Z(1), values), expected)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("seed", range(20))
def test_adjoint_equals_parameter_shift(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 5))
    ops_, values, _ = symbolize(random_circuit(n_wires, depth=20, rng=rng), rng)
    circuit = Circuit(n_wires, ops_)
    observables = _observables(n_wires, rng)
    adjoint = jacobian(circuit, observables, values, method="adjoint", dtype=dtype)
    shift = jacobian(circuit, observables, values, method="parameter-shift", dtype=dtype)
    assert adjoint.shape == (len(observables), len(circuit.parameters))
    assert_close(adjoint, shift, dtype)


def test_deep_circuit_agrees_with_parameter_shift() -> None:
    """Undoing 500 gates one at a time doesn't drift beyond the standard tolerance."""
    rng = np.random.default_rng(7)
    numeric = random_circuit(4, depth=500, rng=rng)
    # Make a few parameters symbolic, spread over the circuit, to keep parameter-shift cheap.
    parametrized = [i for i, op in enumerate(numeric) if op.params]
    chosen = set(parametrized[:: max(1, len(parametrized) // 6)])
    values: dict[str, float] = {}
    symbolic = []
    for i, op in enumerate(numeric):
        if i not in chosen:
            symbolic.append(op)
            continue
        name = f"p{len(values)}"
        values[name] = float(op.params[0])  # type: ignore[arg-type]
        params = (Parameter(name), *op.params[1:])
        symbolic.append(Operation(op.gate, op.wires, params, op.controls, op.control_values))
    circuit = Circuit(4, symbolic)
    observables = _observables(4, rng)
    adjoint = jacobian(circuit, observables, values, method="adjoint")
    shift = jacobian(circuit, observables, values, method="parameter-shift")
    assert_close(adjoint, shift, np.complex128)


def test_gates_before_the_first_trainable_one_still_matter() -> None:
    # H then RY(t): ⟨Z⟩ = -sin t, so d/dt = -cos t. The backward pass stops at RY, but the
    # forward state includes H.
    c = Circuit(1).h(0).ry(t, 0)
    np.testing.assert_allclose(grad(c, Z(0), {"t": 0.4}), [-np.cos(0.4)], atol=1e-12)


# --------------------------------------------------------------------------- reference function


def test_adjoint_jacobian_columns_follow_trainable_order() -> None:
    circuit = [ops.RX(0.3, 0), ops.RY(0.8, 0)]
    forward = adjoint_jacobian(1, circuit, [Z(0)], [(0, 0), (1, 0)])
    backward = adjoint_jacobian(1, circuit, [Z(0)], [(1, 0), (0, 0)])
    np.testing.assert_array_equal(forward[:, ::-1], backward)
    repeated = adjoint_jacobian(1, circuit, [Z(0)], [(0, 0), (0, 0)])
    assert repeated[0, 0] == repeated[0, 1] == forward[0, 0]


def test_adjoint_jacobian_without_trainable_slots() -> None:
    jac = adjoint_jacobian(2, [ops.H(0)], [Z(0), Z(1)], [])
    assert jac.shape == (2, 0)


@pytest.mark.parametrize("slot", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_adjoint_jacobian_rejects_missing_slots(slot: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="doesn't exist"):
        adjoint_jacobian(1, [ops.RX(0.3, 0)], [Z(0)], [slot])


def test_front_end_makes_one_backend_call() -> None:
    """ADR-0004 D4: the whole backward pass is one backend call, not one call per gate."""

    class Counting(NumPyBackend):
        calls = 0
        applied = 0

        def adjoint_jacobian(self, *args: Any, **kwargs: Any) -> Any:
            Counting.calls += 1
            return super().adjoint_jacobian(*args, **kwargs)

        def apply_operation(self, state: Any, op: Operation) -> Any:
            Counting.applied += 1
            return super().apply_operation(state, op)

    backend = Counting()
    assert isinstance(backend, Backend)
    c = Circuit(2).ry(a, 0).cnot(0, 1).crx(b, 0, 1)
    jacobian(c, [Z(0), Z(1)], {"a": 0.1, "b": 0.2}, backend=backend)
    assert (Counting.calls, Counting.applied) == (1, 0)


def test_adjoint_rejects_variance_and_shots() -> None:
    c = Circuit(1).rx(a, 0)
    with pytest.raises(ValueError, match="can't differentiate 'var'; use method='finite-diff'"):
        grad(c, Z(0), {"a": 0.1}, measurement="var")
    with pytest.raises(ValueError, match="use method='parameter-shift'"):
        grad(c, Z(0), {"a": 0.1}, shots=100)


# --------------------------------------------------------------------------- memory


TRAINABLE = 20
"""Trainable parameters in every memory circuit, so only the depth varies."""


def _layered(n_wires: int, depth: int) -> tuple[list[Operation], list[tuple[int, int]]]:
    """``depth`` layers of an RX and a CNOT; ``TRAINABLE`` RX gates, spread evenly, are trainable.

    Holding the number of trainable parameters fixed matters: their bookkeeping (slot lists, the
    Jacobian) is proportional to their number, which is unavoidable and isn't what "memory
    independent of depth" is about.
    """
    circuit: list[Operation] = []
    for d in range(depth):
        w = d % n_wires
        circuit += [ops.RX(0.1 * d, w), ops.CNOT(w, (w + 1) % n_wires)]
    layers = np.linspace(0, depth - 1, TRAINABLE).astype(int)
    return circuit, [(2 * int(d), 0) for d in layers]


def _peak_states(n_wires: int, depth: int, observables: list[Observable]) -> float:
    circuit, trainable = _layered(n_wires, depth)
    # Measure the steady state: a first identical call pays one-time costs (lazily built objects,
    # and Python's free lists filling up, which tracemalloc counts as allocated). Without it the
    # peak depended on which tests had run before.
    adjoint_jacobian(n_wires, circuit, observables, trainable)
    tracemalloc.start()
    try:
        adjoint_jacobian(n_wires, circuit, observables, trainable)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / state_nbytes(n_wires)


def test_adjoint_memory_is_independent_of_depth() -> None:
    """Peak memory is a few states at depth 20 and at depth 500.

    ψ, λ and the scratch μ, plus the kernel's temporaries, measure about 5 states. Storing every
    intermediate state, as reverse-mode autodiff does, would need about 1000 at depth 500.

    Regression: run on its own, the first version failed (6.05 states) yet passed in the full
    suite. It made every layer trainable, so parameter bookkeeping grew with depth, and it measured
    a first call, which pays one-time costs that earlier tests had usually paid already. Measured
    in steady state with a fixed parameter count, the peak stays at 5.1 to 5.35 states up to
    depth 4000.
    """
    n_wires = 14  # one state is 256 KiB, far above incidental allocations
    shallow = _peak_states(n_wires, 20, [Z(0)])
    deep = _peak_states(n_wires, 500, [Z(0)])
    assert shallow <= 6
    assert deep <= 6
    assert abs(deep - shallow) <= 0.25


def test_adjoint_memory_grows_by_one_state_per_observable() -> None:
    """2 + M states: one λ per observable (ADR-0004 D3).

    Regression: updating the λs with a list comprehension held the old and new lists at once,
    so each extra observable cost two states.
    """
    n_wires = 14
    one = _peak_states(n_wires, 50, [Z(0)])
    six = _peak_states(n_wires, 50, [Z(w) for w in range(6)])
    assert 0.8 <= (six - one) / 5 <= 1.2
