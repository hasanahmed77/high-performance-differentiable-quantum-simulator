"""Batching: jax.vmap over a compiled circuit equals a loop, with one compilation (QS-605).

ADR-0007 D1: a batch is ``jax.vmap`` over the compiled function, with no batched kernels.
"""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ketjet.backends import jax_backend
from ketjet.backends.jax_backend import (
    clear_compile_cache,
    compile_circuit,
    expectation,
    parameter_map,
)
from ketjet.backends.numpy_backend import NumPyBackend
from ketjet.circuit import Circuit
from ketjet.observables import Observable
from ketjet.operations import Parameter
from tests.random_circuits import random_circuit, random_observable, symbolize
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
# vmap and a loop run the same program; float32 sums may still be reordered when batched.
GRAD_TOL = {np.complex128: 1e-10, np.complex64: 2e-4}
BATCH = 6


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_compile_cache()
    jax_backend._TRACES.clear()
    yield
    clear_compile_cache()


def _traces() -> int:
    return sum(jax_backend._TRACES.values())


def _problem(seed: int, dtype: Any) -> tuple[Circuit, Observable, jax.Array]:
    """A random symbolic circuit, a random Hamiltonian and a batch of symbol values."""
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 5))
    symbolic, _, _ = symbolize(random_circuit(n_wires, 30, rng), rng)
    circuit = Circuit(n_wires, symbolic)
    real = np.float64 if dtype == np.complex128 else np.float32
    batch = rng.uniform(-np.pi, np.pi, (BATCH, len(circuit.parameters))).astype(real)
    return circuit, random_observable(n_wires, rng), jnp.asarray(batch)


def _state_fn(circuit: Circuit, dtype: Any) -> Callable[[jax.Array], jax.Array]:
    run, to_slots = compile_circuit(circuit, dtype), parameter_map(circuit)
    return lambda x: run(to_slots(x))


# --------------------------------------------------------------------------- results


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(6))
def test_a_batch_of_states_equals_a_loop_and_the_oracle(seed: int, dtype: Any) -> None:
    circuit, _, batch = _problem(seed, dtype)
    state = _state_fn(circuit, dtype)

    batched = jax.vmap(state)(batch)

    assert batched.shape == (BATCH, 2**circuit.n_wires)
    assert batched.dtype == dtype
    assert_close(batched, jnp.stack([state(x) for x in batch]), dtype)
    for row, x in zip(batched, batch, strict=True):
        values = {p.name: float(v) for p, v in zip(circuit.parameters, x, strict=True)}
        oracle = circuit.bind(values).state(backend=NumPyBackend(), dtype=dtype)
        assert_close(row, oracle, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(6))
def test_a_batch_of_gradients_equals_a_loop(seed: int, dtype: Any) -> None:
    circuit, hamiltonian, batch = _problem(seed, dtype)
    state = _state_fn(circuit, dtype)

    def energy(x: jax.Array) -> jax.Array:
        return expectation(state(x), hamiltonian)

    batched = jax.jit(jax.vmap(jax.value_and_grad(energy)))(batch)
    looped = [jax.value_and_grad(energy)(x) for x in batch]

    np.testing.assert_allclose(batched[0], [v for v, _ in looped], atol=GRAD_TOL[dtype])
    np.testing.assert_allclose(batched[1], [g for _, g in looped], atol=GRAD_TOL[dtype])


def test_a_batch_of_jacobians_has_one_row_block_per_entry() -> None:
    circuit, hamiltonian, batch = _problem(0, np.complex128)
    state = _state_fn(circuit, np.complex128)
    observables = [hamiltonian, 2.0 * hamiltonian]

    def energies(x: jax.Array) -> jax.Array:
        s = state(x)
        return jnp.stack([expectation(s, obs) for obs in observables])

    jacobians = jax.vmap(jax.jacrev(energies))(batch)

    assert jacobians.shape == (BATCH, 2, len(circuit.parameters))
    assert_close(jacobians[:, 1], 2.0 * jacobians[:, 0], np.complex128)
    assert_close(jacobians[3], jax.jacrev(energies)(batch[3]), np.complex128)


def test_a_batch_can_share_some_inputs() -> None:
    """``in_axes`` batches one argument and broadcasts the other, as for any JAX function."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circuit = Circuit(1).ry(theta, 0).rz(phi, 0)
    run, to_slots = compile_circuit(circuit), parameter_map(circuit)

    def amplitude(t: jax.Array, p: jax.Array) -> jax.Array:
        return run(to_slots(jnp.stack([t, p])))[1]

    thetas = jnp.linspace(0.0, np.pi, 4)
    amplitudes = jax.vmap(amplitude, in_axes=(0, None))(thetas, jnp.array(0.3))

    # RZ(φ) RY(θ)|0⟩ has amplitude e^{iφ/2} sin(θ/2) on |1⟩.
    assert_close(amplitudes, np.exp(0.15j) * np.sin(np.asarray(thetas) / 2), np.complex128)


# --------------------------------------------------------------------------- compilation


def test_batched_and_single_calls_share_one_compilation() -> None:
    circuit, hamiltonian, batch = _problem(1, np.complex128)
    state = _state_fn(circuit, np.complex128)

    def energy(x: jax.Array) -> jax.Array:
        return expectation(state(x), hamiltonian)

    state(batch[0])
    jax.vmap(state)(batch)
    jax.vmap(state)(batch[:2])  # another batch size
    jax.jit(jax.vmap(jax.grad(energy)))(batch)

    assert _traces() == 1


# --------------------------------------------------------------------------- errors


def test_a_batch_passed_without_vmap_is_refused_with_a_pointer_to_vmap() -> None:
    circuit = Circuit(1).rx(Parameter("a"), 0).ry(Parameter("b"), 0)
    run, to_slots = compile_circuit(circuit), parameter_map(circuit)
    batch = jnp.zeros((5, 2))

    with pytest.raises(ValueError, match=r"takes 2 parameter value\(s\).*use jax\.vmap"):
        run(batch)
    with pytest.raises(ValueError, match=r"has 2 symbol\(s\).*use jax\.vmap"):
        to_slots(batch)


@pytest.mark.parametrize("values", [[0.1], [0.1, 0.2, 0.3], np.zeros((5, 3))])
def test_a_wrong_shape_that_isnt_a_batch_gets_no_vmap_hint(values: Any) -> None:
    run = compile_circuit(Circuit(1).rx(Parameter("a"), 0).ry(Parameter("b"), 0))
    with pytest.raises(ValueError, match="flat array") as error:
        run(values)
    assert "vmap" not in str(error.value)
