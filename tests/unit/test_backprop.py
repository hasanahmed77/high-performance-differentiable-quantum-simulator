"""Backprop: jax.grad through the simulator matches adjoint and parameter-shift (QS-604)."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ketjet import gradients
from ketjet.backends import jax_backend
from ketjet.backends.jax_backend import (
    JaxBackend,
    clear_compile_cache,
    compile_circuit,
    expectation,
    parameter_map,
)
from ketjet.circuit import Circuit
from ketjet.observables import Observable, X, Y, Z
from ketjet.operations import Parameter
from tests.random_circuits import random_circuit, random_observable, symbolize

DTYPES = [np.complex128, np.complex64]
# Gradients of 40-gate circuits: float64 agrees to rounding; float32 accumulates ~1e-6 per gate.
GRAD_TOL = {np.complex128: 1e-10, np.complex64: 2e-4}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_compile_cache()
    jax_backend._TRACES.clear()
    yield
    clear_compile_cache()


def _traces() -> int:
    return sum(jax_backend._TRACES.values())


def _problem(seed: int, n_wires: int = 4, depth: int = 30) -> tuple[Circuit, dict[str, float]]:
    """A random circuit whose every parameter is a symbol, a quarter of them shared."""
    rng = np.random.default_rng(seed)
    symbolic, values, _ = symbolize(random_circuit(n_wires, depth, rng), rng)
    return Circuit(n_wires, symbolic), values


def _energies(circuit: Circuit, observables: list[Observable], dtype: Any = np.complex128) -> Any:
    run, to_slots = compile_circuit(circuit, dtype), parameter_map(circuit)

    def energies(x: jax.Array) -> jax.Array:
        state = run(to_slots(x))
        return jnp.stack([expectation(state, obs) for obs in observables])

    return energies


def _symbol_values(circuit: Circuit, values: dict[str, float], dtype: Any) -> jax.Array:
    real = np.float64 if dtype == np.complex128 else np.float32
    return jnp.asarray([values[p.name] for p in circuit.parameters], dtype=real)


# --------------------------------------------------------------------------- matches the oracle


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(6))
def test_jacrev_matches_the_numpy_adjoint_and_parameter_shift(seed: int, dtype: Any) -> None:
    circuit, values = _problem(seed)
    rng = np.random.default_rng(100 + seed)
    observables = [random_observable(4, rng) for _ in range(3)]

    backprop = jax.jacrev(_energies(circuit, observables, dtype))(
        _symbol_values(circuit, values, dtype)
    )
    adjoint = gradients.jacobian(circuit, observables, values, method="adjoint")
    shift = gradients.jacobian(circuit, observables, values, method="parameter-shift")

    assert backprop.shape == adjoint.shape == (3, len(circuit.parameters))
    np.testing.assert_allclose(backprop, adjoint, atol=GRAD_TOL[dtype], rtol=0)
    np.testing.assert_allclose(backprop, shift, atol=GRAD_TOL[dtype], rtol=0)


@pytest.mark.parametrize("seed", range(4))
def test_grad_of_one_energy_matches_gradients_grad(seed: int) -> None:
    circuit, values = _problem(seed, n_wires=3)
    observable = random_observable(3, np.random.default_rng(seed))
    energy = _energies(circuit, [observable])

    backprop = jax.grad(lambda x: energy(x)[0])(_symbol_values(circuit, values, np.complex128))
    np.testing.assert_allclose(
        backprop, gradients.grad(circuit, observable, values), atol=1e-10, rtol=0
    )


def test_a_shared_symbol_gets_the_sum_of_its_slots_gradients() -> None:
    theta = Parameter("theta")
    # <Z> after RY(θ) three times is cos(3θ): d/dθ = -3 sin(3θ), which needs all three slots.
    circuit = Circuit(1).ry(theta, 0).ry(theta, 0).ry(theta, 0)
    energy = _energies(circuit, [Z(0)])
    assert float(jax.grad(lambda x: energy(x)[0])(jnp.array([0.4]))[0]) == pytest.approx(
        -3 * np.sin(1.2), abs=1e-12
    )


def test_symbols_mixed_with_numeric_parameters() -> None:
    a, b = Parameter("a"), Parameter("b")
    circuit = Circuit(2).rx(a, 0).ry(0.7, 1).cnot(0, 1).rot(b, 0.2, a, 1).crz(-0.4, 1, 0)
    observables: list[Observable] = [Z(0) @ X(1), Y(1)]
    values = {"a": 0.3, "b": -1.1}
    backprop = jax.jacrev(_energies(circuit, observables))(jnp.array([0.3, -1.1]))
    np.testing.assert_allclose(
        backprop, gradients.jacobian(circuit, observables, values), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_gradients_keep_the_circuits_precision(dtype: Any) -> None:
    circuit, values = _problem(0, n_wires=2, depth=10)
    x = _symbol_values(circuit, values, dtype)
    g = jax.grad(lambda v: _energies(circuit, [Z(0)], dtype)(v)[0])(x)
    assert g.dtype == x.dtype


# --------------------------------------------------------------------------- compilation


def test_jit_of_grad_equals_grad_and_compiles_the_circuit_once() -> None:
    circuit, values = _problem(1, n_wires=3, depth=20)
    energy = _energies(circuit, [Z(0) @ Z(2)])
    grad = jax.grad(lambda x: energy(x)[0])
    compiled_grad = jax.jit(grad)
    x0 = _symbol_values(circuit, values, np.complex128)
    for shift in np.linspace(-1, 1, 10):
        np.testing.assert_allclose(compiled_grad(x0 + shift), grad(x0 + shift), atol=1e-12)
    assert _traces() == 1  # the circuit program, shared by grad, jit(grad) and every point


def test_the_jax_backends_adjoint_jacobian_reuses_one_compilation() -> None:
    backend = JaxBackend()
    for seed in range(3):  # same structure, new values
        rng = np.random.default_rng(seed)
        ops = Circuit(2).rx(rng.normal(), 0).cnot(0, 1).ry(rng.normal(), 1).operations
        jac = backend.adjoint_jacobian(2, ops, [Z(1)], [(0, 0), (2, 0), (0, 0)])
        expected = gradients.jacobian(
            Circuit(2).rx(Parameter("a"), 0).cnot(0, 1).ry(Parameter("b"), 1),
            [Z(1)],
            {"a": float(ops[0].params[0]), "b": float(ops[2].params[0])},  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(jac, expected[:, [0, 1, 0]], atol=1e-12)
    assert _traces() == 1


# --------------------------------------------------------------------------- parameter_map


def test_parameter_map_fills_shared_slots_and_keeps_numeric_ones() -> None:
    a, b = Parameter("a"), Parameter("b")
    circuit = Circuit(1).rx(a, 0).ry(0.25, 0).rot(b, a, -0.5, 0)
    np.testing.assert_array_equal(parameter_map(circuit)([1.0, 2.0]), [1.0, 0.25, 2.0, 1.0, -0.5])


def test_parameter_map_follows_the_values_dtype() -> None:
    circuit = Circuit(1).rx(Parameter("a"), 0).ry(0.25, 0)
    assert parameter_map(circuit)(np.array([1.0], dtype=np.float32)).dtype == np.float32


def test_parameter_map_without_symbols_or_parameters() -> None:
    np.testing.assert_array_equal(parameter_map(Circuit(1).rx(0.5, 0))([]), [0.5])
    assert parameter_map(Circuit(1).h(0))([]).shape == (0,)


@pytest.mark.parametrize("values", [[], [0.1, 0.2, 0.3], [[0.1, 0.2]], 0.1])
def test_parameter_map_rejects_the_wrong_number_of_values(values: Any) -> None:
    circuit = Circuit(1).rx(Parameter("a"), 0).ry(Parameter("b"), 0)
    with pytest.raises(ValueError, match=r"has 2 symbol\(s\)"):
        parameter_map(circuit)(values)
