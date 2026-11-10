"""Unit tests for exact measurements and Circuit execution (QS-304, ADR-0003 D2, D4, D6, D7)."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet import Circuit
from ketjet.backends import Backend, NumPyBackend
from ketjet.measurements import expval, probs, var
from ketjet.observables import Hermitian, I, X, Y, Z
from ketjet.operations import Parameter
from ketjet.statevector import StateVector
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
BELL = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)


def random_state(n: int, seed: int, dtype: Any = np.complex128) -> npt.NDArray[Any]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    out: npt.NDArray[Any] = (v / np.linalg.norm(v)).astype(dtype)
    return out


# --------------------------------------------------------------------------- probs


def test_probs_of_bell_state() -> None:
    assert_close(probs(BELL), [0.5, 0, 0, 0.5], np.complex128)
    assert_close(probs(BELL, [0]), [0.5, 0.5], np.complex128)
    assert_close(probs(BELL, [1, 0]), [0.5, 0, 0, 0.5], np.complex128)


def test_probs_wire_order_defines_the_index() -> None:
    # |01> (wire 0 = 0, wire 1 = 1) is index 1 for wires [0, 1] and index 2 for wires [1, 0].
    state = np.array([0, 1, 0, 0], dtype=np.complex128)
    assert_close(probs(state, [0, 1]), [0, 1, 0, 0], np.complex128)
    assert_close(probs(state, [1, 0]), [0, 0, 1, 0], np.complex128)


@pytest.mark.parametrize(
    ("dtype", "real"), [(np.complex128, np.float64), (np.complex64, np.float32)]
)
def test_probs_precision_matches_the_state(dtype: Any, real: Any) -> None:
    p = probs(random_state(4, 0, dtype), [2, 0])
    assert p.dtype == real  # no silent promotion (ADR-0003 D4)
    assert p.flags.c_contiguous


@settings(max_examples=100, deadline=None)
@given(n=st.integers(1, 6), seed=st.integers(0, 2**31), data=st.data())
def test_probs_marginals_are_consistent(n: int, seed: int, data: st.DataObject) -> None:
    psi = random_state(n, seed)
    wires = data.draw(st.permutations(range(n)))[: data.draw(st.integers(1, n))]
    p = probs(psi, wires)
    assert p.shape == (2 ** len(wires),)
    assert_close(p.sum(), 1.0, np.complex128)
    assert (p >= 0).all()
    # Marginalizing the full distribution by hand gives the same answer.
    full = probs(psi).reshape((2,) * n)
    others = tuple(w for w in range(n) if w not in wires)
    expected = np.transpose(full.sum(axis=others), [sorted(wires).index(w) for w in wires])
    assert_close(p, expected.reshape(-1), np.complex128)


@pytest.mark.parametrize(
    ("wires", "error", "message"),
    [([], ValueError, "At least one wire"), ([2], ValueError, "out of range"),
     ([0, 0], ValueError, "distinct"), ([0.0], TypeError, "must be ints")],
)  # fmt: skip
def test_probs_validation(wires: list[Any], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        probs(BELL, wires)


# --------------------------------------------------------------------------- expval and var


@pytest.mark.parametrize(
    ("obs", "mean", "variance"),
    [(Z(0) @ Z(1), 1.0, 0.0), (X(0) @ X(1), 1.0, 0.0), (Y(0) @ Y(1), -1.0, 0.0),
     (Z(0), 0.0, 1.0), (X(1), 0.0, 1.0), (I(), 1.0, 0.0),
     (2 * Z(0) @ Z(1) - 3 * X(0), 2.0, 9.0)],
    ids=repr,
)  # fmt: skip
def test_bell_state_expectations(obs: Any, mean: float, variance: float) -> None:
    assert expval(BELL, obs) == pytest.approx(mean, abs=1e-12)
    assert var(BELL, obs) == pytest.approx(variance, abs=1e-12)


def test_expval_is_linear_in_the_observable() -> None:
    psi = random_state(3, 1)
    a, b = Z(0) @ X(2), Hermitian(np.diag([0.5, -1.5]), [1])
    assert expval(psi, 2 * a - 0.5 * b) == pytest.approx(2 * expval(psi, a) - 0.5 * expval(psi, b))


def test_expval_equals_dense_quadratic_form() -> None:
    psi = random_state(3, 2)
    obs = 0.7 * X(0) @ Y(1) + Hermitian(np.diag([1.0, 2.0, 3.0, 4.0]), [2, 0])
    m = obs.matrix(3)
    assert expval(psi, obs) == pytest.approx(float(np.real(psi.conj() @ m @ psi)))
    assert var(psi, obs) == pytest.approx(
        float(np.real(psi.conj() @ m @ m @ psi)) - expval(psi, obs) ** 2
    )


def test_eigenstate_variance_is_zero_up_to_rounding() -> None:
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    assert 0.0 <= var(plus, X(0)) < 1e-15


def test_negative_rounding_in_the_variance_is_clipped_to_zero() -> None:
    # |psi> is the +1 eigenstate of n.sigma for n = (sin t, 0, cos t). For this t the raw
    # ||O psi||^2 - <O>^2 rounds to -1.1e-16; a variance must never be negative.
    t = 0.01618154538634659
    psi = np.array([np.cos(t / 2), np.sin(t / 2)], dtype=np.complex128)
    obs = float(np.sin(t)) * X(0) + float(np.cos(t)) * Z(0)
    applied = obs.apply(psi)
    raw = float(np.vdot(applied, applied).real) - float(np.vdot(psi, applied).real) ** 2
    assert raw < 0  # the input really does exercise the clip
    assert var(psi, obs) == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_expval_returns_a_python_float(dtype: Any) -> None:
    assert type(expval(random_state(2, 3, dtype), Z(0))) is float
    assert type(var(random_state(2, 3, dtype), Z(0))) is float


# --------------------------------------------------------------------------- backend protocol


def test_backend_measurements_delegate_and_do_not_consume_the_state() -> None:
    backend: Backend[StateVector] = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(2), Circuit(2).h(0).cnot(0, 1))
    before = state.copy()
    assert_close(backend.probs(state), [0.5, 0, 0, 0.5], np.complex128)
    assert backend.expval(state, Z(0) @ Z(1)) == pytest.approx(1.0)
    assert backend.var(state, X(0)) == pytest.approx(1.0)
    np.testing.assert_array_equal(state, before)  # read-only (ADR-0003 D6)


# --------------------------------------------------------------------------- Circuit execution


def test_circuit_measurement_methods() -> None:
    bell = Circuit(2).h(0).cnot(0, 1)
    assert_close(bell.state(), BELL, np.complex128)
    assert_close(bell.probs(), [0.5, 0, 0, 0.5], np.complex128)
    assert_close(bell.probs([1]), [0.5, 0.5], np.complex128)
    assert bell.expval(Z(0) @ Z(1)) == pytest.approx(1.0)
    assert bell.var(Z(0)) == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", ["complex64", np.complex64])
def test_circuit_runs_in_single_precision(dtype: Any) -> None:
    c = Circuit(2).h(0).cnot(0, 1)
    assert c.state(dtype=dtype).dtype == np.complex64
    assert c.probs(dtype=dtype).dtype == np.float32


def test_circuit_accepts_an_explicit_backend() -> None:
    class Counting(NumPyBackend):
        calls = 0

        def apply_operations(self, state: StateVector, ops: Any) -> StateVector:
            Counting.calls += 1
            return super().apply_operations(state, ops)

    Circuit(1).x(0).expval(Z(0), backend=Counting())
    assert Counting.calls == 1


def test_running_an_unbound_circuit_explains_how_to_fix_it() -> None:
    c = Circuit(2).rx(Parameter("theta"), 0).ry(Parameter("phi"), 1)
    for run in (c.state, c.probs, lambda: c.expval(Z(0)), lambda: c.var(Z(0))):
        with pytest.raises(ValueError, match=r"unbound parameter\(s\): theta, phi\. Call bind"):
            run()
    assert c.bind(theta=np.pi, phi=0).expval(Z(0)) == pytest.approx(-1.0)


def test_running_does_not_mutate_the_circuit() -> None:
    c = Circuit(2).h(0)
    c.probs()
    assert len(c) == 1
