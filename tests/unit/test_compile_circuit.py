"""Tests for compile_circuit: one jax.jit compilation per circuit structure (QS-603)."""

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ketjet.backends import jax_backend
from ketjet.backends.jax_backend import (
    COMPILE_CACHE_SIZE,
    JaxBackend,
    circuit_structure,
    clear_compile_cache,
    compile_cache_info,
    compile_circuit,
    parameter_values,
)
from ketjet.backends.numpy_backend import NumPyBackend
from ketjet.circuit import Circuit
from ketjet.operations import Parameter
from tests.random_circuits import random_circuit, symbolize
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """Each test starts with an empty cache and trace count, so its counts are its own."""
    clear_compile_cache()
    jax_backend._TRACES.clear()
    yield
    clear_compile_cache()


def _traces() -> int:
    return sum(jax_backend._TRACES.values())


def _ansatz(n_wires: int, layers: int, theta: Any = 0.1) -> Circuit:
    circuit = Circuit(n_wires)
    for _ in range(layers):
        for w in range(n_wires):
            circuit.ry(theta, w)
        for w in range(n_wires - 1):
            circuit.cnot(w, w + 1)
    return circuit


# --------------------------------------------------------------------------- no retracing


def test_one_structure_traces_once_for_many_parameter_sets() -> None:
    run = compile_circuit(_ansatz(3, 2))
    rng = np.random.default_rng(0)
    for _ in range(25):
        run(rng.uniform(-np.pi, np.pi, run.n_params))
    assert _traces() == 1


def test_a_second_structure_traces_exactly_once_more() -> None:
    first, second = compile_circuit(_ansatz(3, 2)), compile_circuit(_ansatz(3, 3))
    for _ in range(5):
        first(np.zeros(first.n_params))
        second(np.ones(second.n_params))
    assert _traces() == 2


def test_circuits_with_one_structure_share_a_compilation() -> None:
    theta = Parameter("theta")
    # A bound circuit, its symbolic template, and the same structure built again with other values.
    circuits = [_ansatz(2, 2, 0.3), _ansatz(2, 2, theta), _ansatz(2, 2, -1.7)]
    for circuit in circuits:
        compile_circuit(circuit)(np.full(4, 0.5))
    assert _traces() == 1
    assert compile_cache_info().hits == 2


def test_parameter_values_differing_only_in_type_reuse_the_compilation() -> None:
    run = compile_circuit(_ansatz(2, 1))
    run([0.1, 0.2])
    run((0.1, 0.2))
    run(np.array([0.1, 0.2], dtype=np.float32))
    run(jnp.array([1, 2]))  # integers
    assert _traces() == 1


def test_each_precision_is_its_own_compilation() -> None:
    circuit = _ansatz(2, 1)
    for dtype in [*DTYPES, *DTYPES]:
        state = compile_circuit(circuit, dtype)(np.zeros(2))
        assert state.dtype == dtype
    assert _traces() == 2


def test_different_wires_controls_or_control_values_are_different_structures() -> None:
    base = Circuit(3).rx(0.1, 0).cnot(0, 1)
    variants = [
        Circuit(3).rx(0.1, 1).cnot(0, 1),  # target wire
        Circuit(3).rx(0.1, 0).cnot(0, 2),  # gate wires
        Circuit(3).rx(0.1, 0).toffoli(0, 2, 1),  # extra control
        Circuit(4).rx(0.1, 0).cnot(0, 1),  # wire count
        Circuit(3).ry(0.1, 0).cnot(0, 1),  # gate name
    ]
    structures = {circuit_structure(c) for c in [base, *variants]}
    assert len(structures) == 1 + len(variants)


# --------------------------------------------------------------------------- the cache


def test_the_cache_is_bounded_and_drops_the_least_recently_used_program() -> None:
    oldest = _ansatz(1, 1)
    compile_circuit(oldest)
    for layers in range(2, COMPILE_CACHE_SIZE + 2):  # one more structure than the cache holds
        compile_circuit(_ansatz(1, layers))
    assert compile_cache_info().currsize == COMPILE_CACHE_SIZE

    misses = compile_cache_info().misses
    compile_circuit(_ansatz(1, COMPILE_CACHE_SIZE + 1))  # the newest is still there
    assert compile_cache_info().misses == misses
    compile_circuit(oldest)  # the oldest was dropped
    assert compile_cache_info().misses == misses + 1


def test_clearing_the_cache_compiles_again() -> None:
    circuit = _ansatz(2, 1)
    compile_circuit(circuit)(np.zeros(2))
    clear_compile_cache()
    assert compile_cache_info().currsize == 0
    compile_circuit(circuit)(np.zeros(2))
    assert _traces() == 2


# --------------------------------------------------------------------------- results


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(8))
def test_compiled_states_equal_the_eager_backends(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(1, 6))
    circuit = Circuit(n_wires, random_circuit(n_wires, 40, rng))
    state = compile_circuit(circuit, dtype)(parameter_values(circuit))

    assert state.dtype == dtype
    assert_close(state, circuit.state(backend=JaxBackend(), dtype=dtype), dtype)
    assert_close(state, circuit.state(backend=NumPyBackend(), dtype=dtype), dtype)


@pytest.mark.parametrize("seed", range(4))
def test_a_symbolic_template_runs_with_its_bound_values(seed: int) -> None:
    rng = np.random.default_rng(seed)
    symbolic, values, slot_symbol = symbolize(random_circuit(4, 30, rng), rng)
    template = Circuit(4, symbolic)
    bound = template.bind(values)
    slots = [values[f"p{k}"] for k in slot_symbol]  # a shared symbol fills each of its slots

    np.testing.assert_array_equal(parameter_values(bound), slots)
    assert_close(compile_circuit(template)(slots), bound.state(), np.complex128)


def test_a_circuit_without_parameters_takes_no_values() -> None:
    run = compile_circuit(Circuit(2).h(0).cnot(0, 1))
    assert run.n_params == 0
    assert_close(run(), [2**-0.5, 0, 0, 2**-0.5], np.complex128)
    assert_close(compile_circuit(Circuit(1))(), [1, 0], np.complex128)


def test_the_compiled_function_can_be_traced_by_grad_jit_and_vmap() -> None:
    run = compile_circuit(Circuit(1).ry(Parameter("theta"), 0))

    def p1(params: jax.Array) -> jax.Array:
        return jnp.abs(run(params)[1]) ** 2  # sin²(θ/2)

    theta = jnp.array([0.7])
    assert_close(jax.grad(p1)(theta), [np.sin(0.7) / 2], np.complex128)
    assert_close(jax.jit(jax.grad(p1))(theta), [np.sin(0.7) / 2], np.complex128)
    batch = jnp.linspace(0, np.pi, 5)[:, None]
    assert_close(jax.vmap(p1)(batch), np.sin(batch[:, 0] / 2) ** 2, np.complex128)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("params", [[], [0.1], [0.1, 0.2, 0.3], [[0.1, 0.2]], 0.1])
def test_the_wrong_number_or_shape_of_values_is_rejected(params: Any) -> None:
    run = compile_circuit(_ansatz(2, 1))
    with pytest.raises(ValueError, match=r"takes 2 parameter value\(s\) as a flat array"):
        run(params)


def test_parameter_values_needs_a_bound_circuit() -> None:
    with pytest.raises(ValueError, match="unbound parameter"):
        parameter_values(Circuit(1).rx(Parameter("theta"), 0))


def test_complex128_without_64_bit_mode_is_refused_when_compiling_and_when_running() -> None:
    code = (
        "import jax, pytest\n"
        "from ketjet.backends.jax_backend import compile_circuit\n"
        "from ketjet.circuit import Circuit\n"
        "circuit = Circuit(1).rx(0.5, 0)\n"
        "with pytest.raises(RuntimeError, match='jax_enable_x64'):\n"
        "    compile_circuit(circuit)\n"
        "assert compile_circuit(circuit, 'complex64')([0.5]).dtype == 'complex64'\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "run = compile_circuit(circuit)\n"
        "jax.config.update('jax_enable_x64', False)\n"
        "with pytest.raises(RuntimeError, match='jax_enable_x64'):\n"
        "    run([0.5])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
