"""Unit tests for the JAX backend's own functions and rules (QS-602, ADR-0006)."""

import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ketjet import kernels
from ketjet.backends.jax_backend import (
    MATRIX_BUILDERS,
    JaxBackend,
    _contract,
    _scale,
    apply_gate,
    apply_observable,
    expectation,
    gate_matrix,
    probabilities,
    variance,
)
from ketjet.gates import GATES
from ketjet.observables import Hermitian, Observable, Pauli, X, Y, Z
from ketjet.operations import CNOT, RX, RY
from ketjet.statevector import StateVector
from tests.random_circuits import random_observable
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter, where JAX's 64-bit mode is off by default."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def _generic_state(n: int, dtype: Any = np.complex128, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return jnp.asarray(psi / np.linalg.norm(psi), dtype=dtype)


# --------------------------------------------------------------------------- imports (D1)


def test_import_ketjet_does_not_import_jax() -> None:
    code = (
        "import sys, ketjet, ketjet.backends, ketjet.gradients; "
        "ketjet.backends.get_backend('numpy'); print('jax' in sys.modules)"
    )
    out = _python(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------- gate matrices (D3)


def test_every_parametrized_gate_has_a_jax_builder_and_no_other() -> None:
    parametrized = {name for name, spec in GATES.items() if spec.num_params}
    assert set(MATRIX_BUILDERS) == parametrized


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate", sorted(GATES))
def test_jax_matrices_equal_the_numpy_registry(gate: str, dtype: Any) -> None:
    spec = GATES[gate]
    rng = np.random.default_rng(sorted(GATES).index(gate))
    for _ in range(5):
        params = tuple(float(x) for x in rng.uniform(-2 * np.pi, 2 * np.pi, spec.num_params))
        matrix = gate_matrix(gate, params, dtype)
        assert matrix.dtype == np.dtype(dtype)
        assert_close(matrix, spec.matrix(params, dtype), dtype)


def test_gate_matrix_accepts_jax_scalars_and_is_differentiable() -> None:
    def top_left(x: jax.Array) -> jax.Array:
        return gate_matrix("RX", (x,))[0, 0].real  # cos(x / 2)

    assert_close(jax.grad(top_left)(jnp.asarray(0.8)), -0.5 * np.sin(0.4), np.complex128)


def test_gate_matrix_validates_its_arguments() -> None:
    with pytest.raises(KeyError, match="Unknown gate 'RQ'"):
        gate_matrix("RQ", (0.1,))
    with pytest.raises(ValueError, match=r"RX takes 1 parameter\(s\), got 2"):
        gate_matrix("RX", (0.1, 0.2))
    with pytest.raises(ValueError, match="Unsupported dtype"):
        gate_matrix("X", (), "float32")


# --------------------------------------------------------------------------- precision (D5)


def test_complex128_without_64_bit_mode_raises_the_documented_error() -> None:
    code = (
        "from ketjet.backends import get_backend\n"
        "from ketjet.observables import Z\n"
        "from ketjet.operations import RX\n"
        "b = get_backend('jax')\n"
        "for call in (lambda: b.zero_state(2), lambda: b.zero_state(2, 'complex128'),\n"
        "             lambda: b.adjoint_jacobian(1, [RX(0.5, 0)], [Z(0)], [(0, 0)])):\n"
        "    try:\n"
        "        call()\n"
        "    except RuntimeError as exc:\n"
        "        assert 'jax.config.update(\"jax_enable_x64\", True)' in str(exc), exc\n"
        "    else:\n"
        "        raise SystemExit('no error')\n"
        "import jax\n"
        "assert not jax.config.jax_enable_x64  # the backend didn't switch it on\n"
        "s = b.apply_operation(b.zero_state(2, 'complex64'), RX(0.5, 0))\n"
        "assert str(s.dtype) == 'complex64' and b.probs(s).dtype == 'float32'\n"
        "assert abs(b.expval(s, Z(0)) - 0.8775826) < 1e-6\n"
        "jac = b.adjoint_jacobian(1, [RX(0.5, 0)], [Z(0)], [(0, 0)], dtype='complex64')\n"
        "assert jac.dtype == 'float64' and abs(jac[0, 0] + 0.4794255) < 1e-6\n"
        "print('ok')\n"
    )
    out = _python(code)
    assert out.returncode == 0, out.stderr + out.stdout
    assert out.stdout.strip() == "ok"


@pytest.fixture
def without_64_bit_mode() -> Iterator[None]:
    """JAX as a user gets it by default; the suite's setting is restored afterwards."""
    jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


@pytest.mark.usefixtures("without_64_bit_mode")
def test_complex128_is_refused_and_complex64_allowed_without_64_bit_mode() -> None:
    backend = JaxBackend()
    for dtype in (np.complex128, "complex128", np.dtype(np.complex128)):
        with pytest.raises(RuntimeError, match=r'jax.config.update\("jax_enable_x64", True\)'):
            backend.zero_state(2, dtype)
    with pytest.raises(RuntimeError, match="64-bit mode, which is off"):
        backend.zero_state(2)  # complex128 is the default precision
    assert backend.zero_state(2, "complex64").dtype == np.complex64
    assert not jax.config.jax_enable_x64


def test_the_backend_never_changes_the_64_bit_setting() -> None:
    backend = JaxBackend()
    backend.apply_operation(backend.zero_state(2), RX(0.3, 0))
    backend.zero_state(2, "complex64")
    assert jax.config.jax_enable_x64  # still what the test suite set


@pytest.mark.parametrize("dtype", DTYPES)
def test_python_scalars_and_traced_parameters_keep_the_states_precision(dtype: Any) -> None:
    state = JaxBackend().zero_state(2, dtype)
    for params in ((0.4,), (jnp.asarray(0.4),), (jnp.asarray(0.4, dtype=jnp.float32),)):
        out = apply_gate(state, "RY", params, [1], controls=[0], control_values=[False])
        assert out.dtype == np.dtype(dtype)
        assert_close(np.asarray(out)[1], np.sin(0.2), dtype)


# --------------------------------------------------------------------------- kernels (D4)


def test_apply_gate_defaults_to_all_controls_one() -> None:
    state = jnp.zeros(8, dtype=jnp.complex128).at[0b110].set(1)
    out = apply_gate(state, "X", (), [2], controls=[0, 1])
    np.testing.assert_array_equal(np.asarray(out), np.eye(8)[0b111])
    untouched = apply_gate(state, "X", (), [2], controls=[0, 1], control_values=[True, False])
    np.testing.assert_array_equal(np.asarray(untouched), np.asarray(state))


@pytest.mark.parametrize("axes", [(0, 2), (2, 0), (1, 0), (2, 1, 0), (1, 2, 0)])
def test_the_diagonal_kernel_equals_the_dense_one_for_any_axis_order(axes: tuple[int, ...]) -> None:
    """No registry gate is diagonal on two wires yet, so the kernels are compared directly."""
    rng = np.random.default_rng(len(axes) * 10 + axes[0])
    diagonal = jnp.asarray(np.exp(1j * rng.uniform(-3, 3, 2 ** len(axes))))
    psi = _generic_state(3).reshape(2, 2, 2)
    scaled = _scale(psi, diagonal, axes)
    assert_close(scaled, _contract(psi, jnp.diag(diagonal), axes), np.complex128)
    assert_close(
        scaled.reshape(-1),
        kernels.apply_diagonal(np.asarray(psi).reshape(-1), diagonal, axes),
        np.complex128,
    )


def test_apply_gate_validates_wires_and_parameters() -> None:
    state = JaxBackend().zero_state(2)
    with pytest.raises(ValueError, match=r"RX uses wire\(s\) \[2, 3\], but the state has 2 qubit"):
        apply_gate(state, "RX", (0.1,), [3], controls=[2])
    with pytest.raises(ValueError, match=r"RX takes 1 parameter\(s\), got 0"):
        apply_gate(state, "RX", (), [0])
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_gate(state.reshape(2, 2), "X", (), [0])


def test_the_whole_simulation_is_differentiable() -> None:
    """Parameters can be traced end to end: the property QS-604 builds on."""

    def energy(x: jax.Array) -> jax.Array:
        state = JaxBackend().zero_state(2)
        state = apply_gate(state, "RY", (x[0],), [0])
        state = apply_gate(state, "RX", (x[1],), [1], controls=[0])
        return expectation(state, Z(0) @ Z(1) + 0.5 * X(1))

    x = jnp.asarray([0.3, 0.8])
    step = 1e-6
    numeric = [
        (energy(x.at[k].add(step)) - energy(x.at[k].add(-step))) / (2 * step) for k in range(2)
    ]
    np.testing.assert_allclose(np.asarray(jax.grad(energy)(x)), np.asarray(numeric), atol=1e-7)
    assert np.abs(np.asarray(numeric)).min() > 0.01


# --------------------------------------------------------------------------- observables


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "observable",
    [
        Pauli(),
        X(1),
        Y(0),
        Z(2),
        Pauli({0: "Y", 1: "X", 2: "Z"}),
        Y(2) @ Y(0),
        Hermitian([[1, 2 - 1j], [2 + 1j, -0.5]], [1]),
        Hermitian(np.diag([1.0, -2.0, 0.5, 3.0]), [2, 0]) @ Y(1),
        0.7 * X(0) - 1.3 * (Z(1) @ Y(2)) + Hermitian([[0, 1j], [-1j, 0]], [2]),
    ],
    ids=repr,
)
def test_observables_act_like_their_numpy_matrices(observable: Observable, dtype: Any) -> None:
    psi = _generic_state(3, dtype)
    host: StateVector = np.asarray(psi)
    applied = apply_observable(psi, observable)
    assert applied.dtype == np.dtype(dtype)
    assert_close(applied, observable.apply(host), dtype)
    assert_close(applied, observable.matrix(3) @ host, dtype)


@pytest.mark.parametrize("seed", range(10))
def test_expectation_and_variance_match_dense_linear_algebra(seed: int) -> None:
    rng = np.random.default_rng(seed)
    psi = _generic_state(3, seed=seed)
    observable = random_observable(3, rng)
    host, matrix = np.asarray(psi), observable.matrix(3)
    mean = np.vdot(host, matrix @ host).real
    second = np.vdot(host, matrix @ matrix @ host).real
    assert_close(expectation(psi, observable), mean, np.complex128)
    assert_close(variance(psi, observable), second - mean**2, np.complex128)


def test_an_unknown_observable_type_is_rejected_by_name() -> None:
    class Custom(Observable):
        __slots__ = ()

        @property
        def wires(self) -> tuple[int, ...]:
            return (0,)

        def _apply(self, state: StateVector, n_wires: int) -> StateVector:
            return state

    with pytest.raises(TypeError, match="can't apply a Custom observable"):
        apply_observable(JaxBackend().zero_state(1), Custom())


def test_probabilities_follow_the_requested_wire_order() -> None:
    backend = JaxBackend()
    state = backend.apply_operations(backend.zero_state(3), [RY(0.4, 0), RY(1.1, 2), CNOT(0, 1)])
    p0, p2 = [np.cos(0.2) ** 2, np.sin(0.2) ** 2], [np.cos(0.55) ** 2, np.sin(0.55) ** 2]
    assert_close(probabilities(state, [0, 2]), np.kron(p0, p2), np.complex128)
    assert_close(probabilities(state, [2, 0]), np.kron(p2, p0), np.complex128)
    assert_close(probabilities(state).sum(), 1.0, np.complex128)
