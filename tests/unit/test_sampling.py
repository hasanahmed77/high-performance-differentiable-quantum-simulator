"""Tests for seeded sampling and shot-based expectation values (QS-306, ADR-0003 D5).

Statistical tests use fixed seeds, so they're deterministic, and thresholds chosen so that a
correct sampler would fail them with probability below one in a million.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ketjet import Circuit
from ketjet.backends import Backend, NumPyBackend
from ketjet.measurements import estimate_expval, probs, sample, var
from ketjet.observables import Hermitian, I, Observable, X, Y, Z
from ketjet.statevector import StateVector

# Chi-square critical values at significance 1e-6, from scipy.stats.chi2.ppf(1 - 1e-6, df).
CHI2_CRITICAL = {1: 23.928, 3: 30.665, 7: 40.522, 15: 56.493}
BELL = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)


def random_state(n: int, seed: int, dtype: Any = np.complex128) -> npt.NDArray[Any]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    out: npt.NDArray[Any] = (v / np.linalg.norm(v)).astype(dtype)
    return out


def chi_square(bits: npt.NDArray[np.int64], expected_p: npt.NDArray[Any]) -> float:
    k = bits.shape[1]
    index = bits @ (1 << np.arange(k - 1, -1, -1))
    observed = np.bincount(index, minlength=2**k)
    expected = expected_p * len(bits)
    mask = expected > 0
    return float((((observed - expected) ** 2)[mask] / expected[mask]).sum())


# --------------------------------------------------------------------------- reproducibility


def test_same_seed_gives_identical_samples() -> None:
    psi = random_state(4, 0)
    np.testing.assert_array_equal(sample(psi, 500, seed=42), sample(psi, 500, seed=42))
    assert not np.array_equal(sample(psi, 500, seed=42), sample(psi, 500, seed=43))


def test_a_generator_is_used_as_is_and_advances() -> None:
    psi = random_state(3, 1)
    rng = np.random.default_rng(5)
    first, second = sample(psi, 100, seed=rng), sample(psi, 100, seed=rng)
    assert not np.array_equal(first, second)  # the same generator keeps advancing
    rng2 = np.random.default_rng(5)
    np.testing.assert_array_equal(first, sample(psi, 100, seed=rng2))


def test_unseeded_sampling_works() -> None:
    assert sample(BELL, 10).shape == (10, 2)


@pytest.mark.parametrize("wires", [None, [0], [1, 0], [2, 0, 1]])
def test_shape_dtype_and_values(wires: list[int] | None) -> None:
    bits = sample(random_state(3, 2), 64, wires, seed=0)
    assert bits.shape == (64, 3 if wires is None else len(wires))
    assert bits.dtype == np.int64
    assert set(np.unique(bits)) <= {0, 1}


def test_column_order_follows_the_requested_wires() -> None:
    basis_01 = np.array([0, 1, 0, 0], dtype=np.complex128)  # wire 0 = 0, wire 1 = 1
    assert (sample(basis_01, 5, [0, 1], seed=0) == [0, 1]).all()
    assert (sample(basis_01, 5, [1, 0], seed=0) == [1, 0]).all()


def test_bell_samples_are_perfectly_correlated() -> None:
    bits = sample(BELL, 2000, seed=3)
    assert (bits[:, 0] == bits[:, 1]).all()
    assert 0.45 < bits[:, 0].mean() < 0.55


@pytest.mark.parametrize("shots", [0, -1, 1.5, True, "10"])
def test_shots_must_be_a_positive_int(shots: Any) -> None:
    with pytest.raises(ValueError, match="shots must be a positive int"):
        sample(BELL, shots)
    with pytest.raises(ValueError, match="shots must be a positive int"):
        estimate_expval(BELL, Z(0), shots)


def test_complex64_states_sample_without_bias_errors() -> None:
    psi = random_state(6, 4, np.complex64)  # float32 probabilities don't sum to exactly 1
    bits = sample(psi, 20_000, seed=0)
    assert chi_square(bits[:, :3], probs(psi, [0, 1, 2]).astype(np.float64)) < CHI2_CRITICAL[7]


@pytest.mark.parametrize("n", [5, 7, 9])
def test_complex64_uniform_superposition_can_be_sampled(n: int) -> None:
    """Regression: float32 rounding of 1/sqrt(2^n) makes these probabilities sum to 1 - 1.8e-7.

    NumPy's sampler rejects that outright ("Probabilities do not sum to 1"), so without the float64
    renormalization (ADR-0003 D5) sampling one of the most common circuits would crash.
    """
    circuit = Circuit(n)
    for w in range(n):
        circuit.h(w)
    raw = circuit.probs(dtype="complex64").astype(np.float64)
    assert abs(raw.sum() - 1) > 1e-7  # the input really is off by more than NumPy tolerates
    bits = circuit.sample(20_000, [0, 1, 2], seed=0, dtype="complex64")
    assert chi_square(bits, np.full(8, 1 / 8)) < CHI2_CRITICAL[7]


# --------------------------------------------------------------------------- goodness of fit


@pytest.mark.parametrize(("n", "wires"), [(1, None), (2, None), (3, None), (4, None), (4, [3, 1])])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sample_frequencies_match_probs_chi_square(
    n: int, wires: list[int] | None, seed: int
) -> None:
    psi = random_state(n, 100 + seed)
    bits = sample(psi, 50_000, wires, seed=seed)
    expected = probs(psi, wires).astype(np.float64)
    assert chi_square(bits, expected) < CHI2_CRITICAL[expected.size - 1]


# --------------------------------------------------------------------------- shot-based expvals


def sigma(state: StateVector, obs: Observable, shots: int) -> float:
    """Standard error of the shot estimate (independent sampling per Sum term)."""
    terms = obs.terms if hasattr(obs, "terms") else ((1.0, obs),)
    return float(np.sqrt(sum(c * c * var(state, t) for c, t in terms) / shots))


OBSERVABLES = [
    Z(0),
    X(1),
    Y(2),
    Z(0) @ X(1) @ Y(2),
    Hermitian(np.array([[2.0, 1 - 1j], [1 + 1j, -0.5]]), [1]),
    Hermitian(np.diag([1.0, -2.0, 3.0, 0.5]), [2, 0]) @ Y(1),
    0.5 * Z(0) @ Z(1) - 0.7 * X(2) + 0.3 * Y(0) @ X(1),
    1.5 * I() - Z(2),
]


@pytest.mark.parametrize("obs", OBSERVABLES, ids=repr)
@pytest.mark.parametrize("seed", [0, 1])
def test_shot_estimate_is_within_four_sigma_of_exact(obs: Observable, seed: int) -> None:
    psi = Circuit(3).h(0).ry(0.8, 1).cnot(0, 2).crx(1.1, 1, 2).s(0).state()
    shots = 20_000
    exact = float(np.vdot(psi, obs.apply(psi)).real)
    estimate = estimate_expval(psi, obs, shots, seed=seed)
    tolerance = 4 * sigma(psi, obs, shots) + 1e-12
    assert abs(estimate - exact) <= tolerance


def test_eigenstates_are_estimated_exactly() -> None:
    assert estimate_expval(BELL, Z(0) @ Z(1), 100, seed=0) == 1.0
    assert estimate_expval(BELL, Y(0) @ Y(1), 100, seed=0) == -1.0
    assert estimate_expval(BELL, 2 * I(), 10, seed=0) == 2.0


def test_shot_estimate_is_reproducible_and_converges() -> None:
    psi = random_state(3, 9)
    obs = X(0) @ Z(2)
    assert estimate_expval(psi, obs, 1000, seed=1) == estimate_expval(psi, obs, 1000, seed=1)
    exact = float(np.vdot(psi, obs.apply(psi)).real)
    errors = [abs(estimate_expval(psi, obs, s, seed=2) - exact) for s in (100, 100_000)]
    assert errors[1] < errors[0]


def test_shot_estimate_rejects_observables_outside_the_state() -> None:
    with pytest.raises(ValueError, match=r"acts on wire\(s\) \[5\]"):
        estimate_expval(BELL, Z(5), 10, seed=0)


# --------------------------------------------------------------------------- backend and Circuit


def test_backend_sample_delegates_and_is_read_only() -> None:
    backend: Backend[StateVector] = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(2), Circuit(2).h(0).cnot(0, 1))
    before = state.copy()
    bits = backend.sample(state, 100, seed=1)
    assert (bits[:, 0] == bits[:, 1]).all()
    np.testing.assert_array_equal(state, before)


def test_circuit_sample_and_shot_expval() -> None:
    bell = Circuit(2).h(0).cnot(0, 1)
    np.testing.assert_array_equal(bell.sample(50, seed=4), sample(BELL, 50, seed=4))
    assert bell.sample(10, [1], seed=0).shape == (10, 1)
    assert bell.expval(Z(0) @ Z(1), shots=100, seed=0) == 1.0
    assert abs(bell.expval(X(0), shots=10_000, seed=0)) < 4 * np.sqrt(1 / 10_000)
