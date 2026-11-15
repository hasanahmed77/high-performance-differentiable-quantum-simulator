"""Cross-backend differential tests: every fast backend equals the NumPy oracle (QS-602).

Random circuits over every registry gate, with random controls and control values, give generic
(asymmetric) states. Each backend method is compared with the NumPy backend's at the standard
tolerances, in both precisions (plan.md §6.1, §6.3).
"""

from typing import Any

import numpy as np
import pytest

from ketjet.backends import Backend, NumPyBackend, get_backend
from ketjet.observables import Z
from tests.random_circuits import random_circuit, random_observable
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
ORACLE = NumPyBackend()


@pytest.fixture(params=["jax"])
def fast(request: pytest.FixtureRequest) -> Backend[Any]:
    """Every backend other than the oracle."""
    pytest.importorskip(request.param)
    return get_backend(request.param)


def _case(seed: int) -> tuple[int, list[Any], np.random.Generator]:
    rng = np.random.default_rng(6020 + seed)
    n = int(rng.integers(2, 7))
    return n, random_circuit(n, depth=30, rng=rng), rng


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(25))
def test_states_match_the_oracle(fast: Backend[Any], seed: int, dtype: Any) -> None:
    n, circuit, _ = _case(seed)
    state = fast.apply_operations(fast.zero_state(n, dtype), circuit)
    expected = ORACLE.apply_operations(ORACLE.zero_state(n, dtype), circuit)
    assert state.dtype == np.dtype(dtype)  # never promoted along the way
    assert fast.to_numpy(state).dtype == np.dtype(dtype)
    assert_close(fast.to_numpy(state), expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(25))
def test_measurements_match_the_oracle(fast: Backend[Any], seed: int, dtype: Any) -> None:
    n, circuit, rng = _case(seed)
    state = fast.apply_operations(fast.zero_state(n, dtype), circuit)
    expected = ORACLE.apply_operations(ORACLE.zero_state(n, dtype), circuit)

    subset = [int(w) for w in rng.permutation(n)[: int(rng.integers(1, n + 1))]]
    for wires in (None, subset):
        p = fast.probs(state, wires)
        assert p.dtype == ORACLE.probs(expected, wires).dtype
        assert_close(p, ORACLE.probs(expected, wires), dtype)

    observable = random_observable(n, rng)
    assert_close(fast.expval(state, observable), ORACLE.expval(expected, observable), dtype)
    # A variance is a difference of two O(|O|²) numbers, so compare relative to that scale.
    scale = float(np.linalg.norm(observable.matrix(n), 2)) ** 2
    assert_close(
        fast.var(state, observable) / scale, ORACLE.var(expected, observable) / scale, dtype
    )


@pytest.mark.parametrize("seed", range(10))
def test_a_seed_gives_the_oracles_samples(fast: Backend[Any], seed: int) -> None:
    """One seed, the same bits on every backend (ADR-0006 D6).

    In complex128 the two backends' probabilities differ by about 1e-16, far below the spacing a
    draw could land in. complex64 is covered statistically by the device tests.
    """
    n, circuit, rng = _case(seed)
    state = fast.apply_operations(fast.zero_state(n), circuit)
    expected = ORACLE.apply_operations(ORACLE.zero_state(n), circuit)
    subset = [int(w) for w in rng.permutation(n)[: int(rng.integers(1, n + 1))]]
    for wires in (None, subset):
        np.testing.assert_array_equal(
            fast.sample(state, 500, wires, seed=seed), ORACLE.sample(expected, 500, wires, seed)
        )
    generator = np.random.default_rng(seed)
    first = fast.sample(state, 20, seed=generator)
    second = fast.sample(state, 20, seed=generator)  # a Generator is used as is, and advances
    reference = np.random.default_rng(seed)
    np.testing.assert_array_equal(first, ORACLE.sample(expected, 20, seed=reference))
    np.testing.assert_array_equal(second, ORACLE.sample(expected, 20, seed=reference))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(15))
def test_adjoint_jacobians_match_the_oracle(fast: Backend[Any], seed: int, dtype: Any) -> None:
    n, circuit, rng = _case(seed)
    slots = [(i, j) for i, op in enumerate(circuit) for j in range(len(op.params))]
    chosen = [slots[int(k)] for k in rng.permutation(len(slots))[:8]]
    chosen.append(chosen[0])  # a slot named twice gets two equal columns
    observables = [random_observable(n, rng), Z(0)]

    jac = fast.adjoint_jacobian(n, circuit, observables, chosen, dtype=dtype)
    expected = ORACLE.adjoint_jacobian(n, circuit, observables, chosen, dtype=dtype)
    assert jac.dtype == np.float64
    assert jac.shape == (2, len(chosen))
    assert np.abs(expected).max() > 1e-2  # the comparison isn't between zeros
    # Derivatives scale with the observable, as its expectation value does.
    scale = max(1.0, float(np.linalg.norm(observables[0].matrix(n), 2)))
    assert_close(jac / scale, expected / scale, dtype)
    np.testing.assert_array_equal(jac[:, 0], jac[:, -1])
