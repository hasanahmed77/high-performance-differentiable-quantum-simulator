"""Adjoint inside JAX: expval_fn's custom_vjp equals backprop, in depth-independent memory (QS-606).

ADR-0007 D2: ``diff_method="adjoint"`` is the adjoint recursion of ADR-0004 as a
``jax.custom_vjp``; one backward pass serves every observable (QS-405).
"""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ketjet import gradients
from ketjet.backends import jax_backend
from ketjet.backends.jax_backend import JaxBackend, clear_compile_cache, expval_fn
from ketjet.circuit import Circuit
from ketjet.observables import Observable, X, Y, Z
from ketjet.operations import Operation, Parameter
from tests.random_circuits import random_circuit, random_observable, symbolize

DTYPES = [np.complex128, np.complex64]
# The same tolerances as backprop against the NumPy adjoint (test_backprop.py): float32 undoes
# every gate on the way back, so it accumulates ~1e-6 per gate.
GRAD_TOL = {np.complex128: 1e-10, np.complex64: 2e-4}
BATCH = 4


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    clear_compile_cache()
    jax_backend._TRACES.clear()
    jax_backend._UNDONE.clear()
    yield
    clear_compile_cache()


def _undone() -> int:
    return sum(jax_backend._UNDONE.values())


def _real(dtype: Any) -> Any:
    return np.float64 if dtype == np.complex128 else np.float32


def _problem(
    seed: int, dtype: Any, n_observables: int = 2
) -> tuple[Circuit, list[Observable], dict[str, float], jax.Array]:
    """A random symbolic circuit with controls and shared symbols, random Hamiltonians, values."""
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 5))
    symbolic, values, _ = symbolize(random_circuit(n_wires, 30, rng), rng)
    circuit = Circuit(n_wires, symbolic)
    observables = [random_observable(n_wires, rng) for _ in range(n_observables)]
    x = jnp.asarray([values[p.name] for p in circuit.parameters], dtype=_real(dtype))
    return circuit, observables, values, x


# --------------------------------------------------------------------------- values


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(6))
def test_adjoint_equals_backprop_and_the_numpy_adjoint(seed: int, dtype: Any) -> None:
    circuit, observables, values, x = _problem(seed, dtype)
    adjoint = expval_fn(circuit, observables, dtype=dtype, diff_method="adjoint")
    backprop = expval_fn(circuit, observables, dtype=dtype)

    np.testing.assert_allclose(adjoint(x), backprop(x), atol=GRAD_TOL[dtype])
    jac = jax.jacrev(adjoint)(x)
    np.testing.assert_allclose(jac, jax.jacrev(backprop)(x), atol=GRAD_TOL[dtype], rtol=0)
    numpy_adjoint = gradients.jacobian(circuit, observables, values, dtype=dtype)
    np.testing.assert_allclose(jac, numpy_adjoint, atol=GRAD_TOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_adjoint_under_jit_and_vmap_equals_a_loop(dtype: Any) -> None:
    circuit, observables, _, x = _problem(3, dtype)
    f = expval_fn(circuit, observables, dtype=dtype, diff_method="adjoint")

    def energy(v: jax.Array) -> jax.Array:
        return jnp.sum(f(v))

    rng = np.random.default_rng(0)
    batch = jnp.asarray(x + rng.normal(0, 0.3, (BATCH, len(x))), dtype=_real(dtype))
    batched = jax.jit(jax.vmap(jax.value_and_grad(energy)))(batch)
    looped = [jax.value_and_grad(energy)(v) for v in batch]

    np.testing.assert_allclose(batched[0], [v for v, _ in looped], atol=GRAD_TOL[dtype])
    np.testing.assert_allclose(batched[1], [g for _, g in looped], atol=GRAD_TOL[dtype])


def test_a_vjp_is_the_gradient_of_the_weighted_sum() -> None:
    """ADR-0004 D6: the cotangent v turns M observables into the one observable Σ v_m O_m."""
    circuit, observables, values, x = _problem(1, np.complex128, n_observables=3)
    f = expval_fn(circuit, observables, diff_method="adjoint")
    v = jnp.array([0.5, -2.0, 1.25])

    _, pullback = jax.vjp(f, x)
    expected = gradients.vjp(circuit, observables, values, v)
    np.testing.assert_allclose(pullback(v)[0], expected, atol=1e-10, rtol=0)


def test_every_parametrized_gate_and_controls() -> None:
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    circuit = (
        Circuit(3)
        .h(0)
        .crx(a, 0, 1)
        .rz(b, 2)
        .phase(c, 1)
        .rot(a, b, c, 2)
        .u3(c, a, b, 0)
        .cry(b, 2, 0)
        .toffoli(0, 1, 2)
    )
    observables = [Z(0) @ X(2), Y(1)]
    x = jnp.array([0.3, -0.7, 1.1])
    np.testing.assert_allclose(
        jax.jacrev(expval_fn(circuit, observables, diff_method="adjoint"))(x),
        jax.jacrev(expval_fn(circuit, observables))(x),
        atol=1e-10,
        rtol=0,
    )


def test_numeric_parameters_stay_fixed() -> None:
    circuit = Circuit(1).ry(0.4, 0).rx(Parameter("a"), 0)  # <Z> = cos(0.4) cos(a)
    grad = jax.grad(lambda v: expval_fn(circuit, [Z(0)], diff_method="adjoint")(v)[0])
    assert float(grad(jnp.array([0.9]))[0]) == pytest.approx(-np.cos(0.4) * np.sin(0.9))


def test_no_observables_and_no_symbols() -> None:
    circuit = Circuit(1).rx(Parameter("a"), 0)
    assert expval_fn(circuit, [], diff_method="adjoint")(jnp.array([0.1])).shape == (0,)
    fixed = expval_fn(Circuit(1).rx(0.5, 0), [Z(0)], diff_method="adjoint")
    assert jax.jacrev(fixed)(jnp.zeros(0)).shape == (1, 0)
    assert float(fixed(jnp.zeros(0))[0]) == pytest.approx(np.cos(0.5))


# --------------------------------------------------------------------------- limits and errors


def test_adjoint_has_no_forward_mode_but_has_a_hessian() -> None:
    """A custom_vjp has no JVP, but forward mode over its backward pass is the Hessian."""
    circuit, observables, _, x = _problem(2, np.complex128, n_observables=1)
    adjoint = expval_fn(circuit, observables, diff_method="adjoint")
    backprop = expval_fn(circuit, observables)

    with pytest.raises(TypeError, match="forward-mode"):
        jax.jacfwd(adjoint)(x)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(adjoint, (x,), (x,))
    np.testing.assert_allclose(
        jax.hessian(lambda v: adjoint(v)[0])(x),
        jax.hessian(lambda v: backprop(v)[0])(x),
        atol=1e-10,
        rtol=0,
    )


def test_errors() -> None:
    circuit = Circuit(1).rx(Parameter("a"), 0)
    with pytest.raises(ValueError, match="Unknown diff_method 'parameter-shift'"):
        expval_fn(circuit, [Z(0)], diff_method="parameter-shift")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wire"):
        expval_fn(circuit, [Z(1)], diff_method="adjoint")
    with pytest.raises(ValueError, match=r"has 1 symbol\(s\)"):
        expval_fn(circuit, [Z(0)], diff_method="adjoint")(jnp.zeros(2))


# --------------------------------------------------------------------------- one pass (QS-405)


def _one_pass_problem(n_observables: int) -> tuple[Circuit, list[Observable], dict[str, float]]:
    rng = np.random.default_rng(5)
    symbolic, values, _ = symbolize(random_circuit(3, 25, rng), rng)
    observables = [random_observable(3, rng) for _ in range(n_observables)]
    return Circuit(3, symbolic), observables, values


def test_many_observables_share_one_backward_pass_on_jax() -> None:
    """The backward rule is traced once and undoes each gate once, for 1 or 8 observables."""
    undone = []
    for n_observables in (1, 8):
        clear_compile_cache()
        jax_backend._UNDONE.clear()
        circuit, observables, values = _one_pass_problem(n_observables)
        x = jnp.asarray([values[p.name] for p in circuit.parameters])
        jac = jax.jacrev(expval_fn(circuit, observables, diff_method="adjoint"))(x)
        assert jac.shape == (n_observables, len(circuit.parameters))
        undone.append(_undone())
    assert undone == [len(circuit.operations)] * 2


def test_many_observables_share_one_backward_pass_on_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sprint 3's NumPy adjoint builds each gate's matrix as often for 8 observables as for 1."""
    built = 0
    matrix = Operation.matrix

    def counting(self: Operation, *args: Any, **kwargs: Any) -> Any:
        nonlocal built
        built += 1
        return matrix(self, *args, **kwargs)

    monkeypatch.setattr(Operation, "matrix", counting)
    counts = []
    for n_observables in (1, 8):
        built = 0
        gradients.jacobian(*_one_pass_problem(n_observables))
        counts.append(built)
    n_ops = len(_one_pass_problem(1)[0].operations)
    assert counts[0] == counts[1] <= 2 * n_ops  # one forward and one backward pass


# --------------------------------------------------------------------------- the backend


def test_the_jax_backends_adjoint_jacobian_runs_the_backward_pass_once_per_structure() -> None:
    backend = JaxBackend()
    for seed in range(3):  # same structure, new values
        rng = np.random.default_rng(seed)
        ops = Circuit(2).rx(rng.normal(), 0).cnot(0, 1).ry(rng.normal(), 1).operations
        jac = backend.adjoint_jacobian(2, ops, [Z(1), X(0)], [(0, 0), (2, 0)])
        expected = gradients.jacobian(
            Circuit(2).rx(Parameter("a"), 0).cnot(0, 1).ry(Parameter("b"), 1),
            [Z(1), X(0)],
            {"a": float(ops[0].params[0]), "b": float(ops[2].params[0])},  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(jac, expected, atol=1e-12)
    assert _undone() == 3  # one trace of the three-gate backward pass


# --------------------------------------------------------------------------- memory


def _layers(n_wires: int, depth: int) -> Circuit:
    """``depth`` layers of an RY on every wire and a CNOT ladder, one symbol per RY."""
    circuit = Circuit(n_wires)
    for d in range(depth):
        for w in range(n_wires):
            circuit = circuit.ry(Parameter(f"t{d}_{w}"), w)
        for w in range(n_wires - 1):
            circuit = circuit.cnot(w, w + 1)
    return circuit


_JAX_VERSION = tuple(int(part) for part in jax.__version__.split(".")[:2])


def _temp_states(n_wires: int, depth: int, diff_method: Any) -> float:
    """XLA's temporary memory for one compiled gradient, in statevectors (ADR-0007 D2)."""
    circuit = _layers(n_wires, depth)
    f = expval_fn(circuit, [Z(0)], diff_method=diff_method)
    grad = jax.jit(jax.grad(lambda v: f(v)[0]))
    x = jnp.zeros(len(circuit.parameters))
    analysis = grad.lower(x).compile().memory_analysis()
    assert analysis is not None
    temp: int = analysis.temp_size_in_bytes
    return temp / (16 << n_wires)


def test_adjoint_memory_is_independent_of_depth_and_backprops_is_not() -> None:
    """At 12 qubits the adjoint gradient needs ~5 states at any depth; backprop one per RY gate.

    The slot array and the gradient grow with depth too, by ~2 kB per layer here: 3% of a state.
    JAX 0.10's XLA doesn't keep the barriers' schedule flat: there the adjoint grows by ~0.4
    states per layer, against backprop's ~12, so only that ratio is checked (ADR-0007 D2).
    """
    shallow, deep = 4, 12
    adjoint = [_temp_states(12, depth, "adjoint") for depth in (shallow, deep)]
    backprop = [_temp_states(12, depth, "backprop") for depth in (shallow, deep)]
    adjoint_growth = adjoint[1] - adjoint[0]
    backprop_growth = backprop[1] - backprop[0]

    assert backprop_growth > 6 * (deep - shallow)  # 12 RY gates per layer
    if _JAX_VERSION < (0, 11):
        assert adjoint_growth < backprop_growth / 10
    else:
        assert adjoint[1] < 8
        assert adjoint_growth < 1
