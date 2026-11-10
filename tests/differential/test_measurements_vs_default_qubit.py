"""Differential tests: probs, expval and var equal PennyLane's default.qubit (QS-304, ADR-0003)."""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.observables import Hermitian, I, Observable, Pauli, X, Y, Z
from tests.tolerances import assert_close

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

from tests.pennylane_support import (  # noqa: E402
    random_circuit,
    run_default_qubit,
    to_pennylane_observable,
)

DTYPES = [np.complex128, np.complex64]
SEEDS = range(20)


def random_observable(n: int, rng: np.random.Generator) -> Observable:
    """A random Hamiltonian mixing Pauli words, a Hermitian factor and an identity offset."""
    terms: Observable = 0.0 * I()
    for _ in range(int(rng.integers(1, 4))):
        word = {
            w: "XYZ"[int(rng.integers(3))]
            for w in rng.permutation(n)[: int(rng.integers(1, n + 1))]
        }
        terms = terms + float(rng.uniform(-2, 2)) * Pauli({int(k): v for k, v in word.items()})
    wire = int(rng.integers(n))
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    return terms + float(rng.uniform(-1, 1)) * Hermitian((a + a.conj().T) / 2, [wire])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_probs_on_random_wire_subsets(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    ops = random_circuit(n, depth=20, rng=rng)
    wires = [int(w) for w in rng.permutation(n)[: int(rng.integers(1, n + 1))]]
    ours = Circuit(n, ops).probs(wires, dtype=dtype)
    expected = run_default_qubit(ops, n, lambda: qml.probs(wires=wires))
    assert ours.dtype == (np.float64 if dtype == np.complex128 else np.float32)
    assert_close(ours, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_expval_and_var_of_random_hamiltonians(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 6))
    ops = random_circuit(n, depth=20, rng=rng)
    obs = random_observable(n, rng)
    circuit = Circuit(n, ops)
    pl_obs = to_pennylane_observable(obs)
    assert_close(
        circuit.expval(obs, dtype=dtype),
        run_default_qubit(ops, n, lambda: qml.expval(pl_obs)),
        dtype,
    )
    assert_close(
        circuit.var(obs, dtype=dtype), run_default_qubit(ops, n, lambda: qml.var(pl_obs)), dtype
    )


@pytest.mark.parametrize(
    "obs",
    [
        Z(0),
        X(1),
        Y(2),
        Z(0) @ Z(1) @ Z(2),
        X(0) @ Y(2),
        Hermitian(np.diag([1.0, 2.0, 3.0, 4.0]), [2, 0]),
    ],
    ids=repr,
)
def test_named_observables_on_a_fixed_circuit(obs: Observable) -> None:
    ops = list(Circuit(3).h(0).ry(0.4, 1).cnot(0, 2).crx(0.7, 1, 2).s(0))
    pl_obs = to_pennylane_observable(obs)
    assert_close(
        Circuit(3, ops).expval(obs),
        run_default_qubit(ops, 3, lambda: qml.expval(pl_obs)),
        np.complex128,
    )
    assert_close(
        Circuit(3, ops).var(obs), run_default_qubit(ops, 3, lambda: qml.var(pl_obs)), np.complex128
    )


def test_state_matches_default_qubit() -> None:
    rng = np.random.default_rng(7)
    ops = random_circuit(4, depth=25, rng=rng)
    assert_close(Circuit(4, ops).state(), run_default_qubit(ops, 4, qml.state), np.complex128)
