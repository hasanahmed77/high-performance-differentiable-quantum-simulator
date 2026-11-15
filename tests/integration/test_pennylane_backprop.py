"""Backprop through ketjet.qubit on the JAX backend, at the device level (QS-608, ADR-0006 D8).

These tests call ``KetjetQubit.execute`` directly with JAX values, without PennyLane's JAX
interface, so they run on every JAX version the backend supports. The QNode-level tests, which
need PennyLane's JAX interface and so JAX < 0.11, are in ``test_pennylane_jax_interface.py``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
jax = pytest.importorskip("jax")
from tests.pennylane_support import PENNYLANE_JAX_DEPRECATIONS  # noqa: E402

pytestmark = [
    pytest.mark.pennylane,
    pytest.mark.filterwarnings(PENNYLANE_JAX_DEPRECATIONS),  # PennyLane 0.45 on JAX 0.10
]

import jax.numpy as jnp  # noqa: E402
from pennylane.devices import ExecutionConfig  # noqa: E402
from pennylane.exceptions import DeviceError, QuantumFunctionError  # noqa: E402
from pennylane.math import Interface  # noqa: E402
from pennylane.tape import QuantumScript  # noqa: E402

from ketjet.backends import jax_backend  # noqa: E402
from ketjet.pennylane import KetjetQubit  # noqa: E402
from tests.pennylane_support import PENNYLANE_JAX_ONLY, PENNYLANE_JAX_WORKS  # noqa: E402

BACKPROP = ExecutionConfig(gradient_method="backprop", interface=Interface.JAX)
WIRES = ["a", 1, "c"]  # labels, not 0..n-1, so a wire mapping mistake shows


def _ops(x: Any) -> list[Any]:
    """Supported gates with controls and an asymmetric state, parametrized by x[0..3]."""
    return [
        qml.RX(x[0], "a"),
        qml.RY(-0.7, 1),
        qml.CNOT(["a", 1]),
        qml.CRZ(x[1], [1, "c"]),
        qml.Rot(x[2], 0.4, x[0], "c"),
        qml.ctrl(qml.RY(x[3], "a"), control=["c"], control_values=[0]),
        qml.GlobalPhase(x[1]),
    ]


def _execute(device: Any, measurements: list[Any], x: Any, config: Any = BACKPROP) -> Any:
    return device.execute([QuantumScript(_ops(x), measurements)], config)[0]


X0 = jnp.array([0.3, -1.2, 0.8, 0.5])


@pytest.fixture
def device() -> Any:
    return KetjetQubit(wires=WIRES, backend="jax")


@pytest.fixture
def reference() -> Any:
    return qml.device("default.qubit", wires=WIRES)


# --------------------------------------------------------------------------- support rules


def test_backprop_is_supported_on_the_jax_backend_only_and_without_shots() -> None:
    exact = QuantumScript([qml.RX(0.1, 0)], [qml.expval(qml.PauliZ(0))])
    shots = QuantumScript([qml.RX(0.1, 0)], [qml.expval(qml.PauliZ(0))], shots=100)
    jax_device, numpy_device = KetjetQubit(backend="jax"), KetjetQubit()
    assert jax_device.supports_derivatives(BACKPROP)
    assert jax_device.supports_derivatives(BACKPROP, exact)
    assert not jax_device.supports_derivatives(BACKPROP, shots)
    assert not numpy_device.supports_derivatives(BACKPROP)
    assert not numpy_device.supports_derivatives(BACKPROP, exact)


def test_backprop_on_the_numpy_backend_raises_pennylanes_error() -> None:
    with pytest.raises(QuantumFunctionError, match="does not support backprop"):
        qml.QNode(lambda: qml.expval(qml.PauliZ(0)), KetjetQubit(wires=1), diff_method="backprop")


def test_backprop_with_an_interface_jax_cant_trace_is_refused() -> None:
    config = ExecutionConfig(gradient_method="backprop", interface=Interface.AUTOGRAD)
    with pytest.raises(DeviceError, match="needs a JAX interface"):
        KetjetQubit(backend="jax").setup_execution_config(config)


def test_a_tape_with_shots_is_refused_under_backprop(device: Any) -> None:
    tape = QuantumScript(_ops(X0), [qml.expval(qml.PauliZ("a"))], shots=100)
    with pytest.raises(DeviceError, match="needs exact results"):
        device.execute([tape], BACKPROP)


# --------------------------------------------------------------------------- values and gradients


MEASUREMENTS: dict[str, Callable[[], Any]] = {
    "expval Pauli word": lambda: qml.expval(qml.PauliZ("a") @ qml.PauliX("c")),
    "expval Hamiltonian": lambda: qml.expval(
        0.4 * qml.PauliY(1) - 1.3 * qml.PauliZ("a") @ qml.PauliZ("c") + 0.2 * qml.Identity(1)
    ),
    "expval Hermitian": lambda: qml.expval(
        qml.Hermitian(np.array([[1.0, 0.5 - 0.2j], [0.5 + 0.2j, -0.3]]), wires="c")
    ),
    "var": lambda: qml.var(qml.PauliX(1) @ qml.PauliY("c")),
    "probs, reordered wires": lambda: qml.probs(wires=["c", "a"]),
    "probs of an observable (fallback)": lambda: qml.probs(op=qml.PauliX(1)),
    "density matrix (fallback)": lambda: qml.density_matrix(wires=["c"]),
}
NEEDS_PENNYLANE_JAX = {"density matrix (fallback)"}  # PennyLane's own code calls is_concrete


def _real_output(result: Any) -> Any:
    """Reduce any measurement to a real scalar, so it can be differentiated by jax.grad."""
    flat = jnp.ravel(jnp.asarray(result))
    weights = jnp.arange(1, flat.size + 1)
    return jnp.sum(weights * (flat.real + 0.5 * flat.imag))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(not PENNYLANE_JAX_WORKS, reason=PENNYLANE_JAX_ONLY)
            if name in NEEDS_PENNYLANE_JAX
            else (),
        )
        for name in MEASUREMENTS
    ],
)
def test_values_and_gradients_match_default_qubit(name: str, device: Any, reference: Any) -> None:
    measurement = MEASUREMENTS[name]

    def ours(x: Any) -> Any:
        return _real_output(_execute(device, [measurement()], x))

    def theirs(x: Any) -> Any:
        return _real_output(_execute(reference, [measurement()], x))

    np.testing.assert_allclose(ours(X0), theirs(X0), atol=1e-10)
    np.testing.assert_allclose(jax.grad(ours)(X0), jax.grad(theirs)(X0), atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(ours))(X0), jax.grad(theirs)(X0), atol=1e-10)


def test_several_measurements_and_the_state_with_its_global_phase(
    device: Any, reference: Any
) -> None:
    measurements = [qml.expval(qml.PauliZ(1)), qml.state(), qml.probs(wires=[1])]

    def ours(x: Any) -> Any:
        return _execute(device, measurements, x)

    def theirs(x: Any) -> Any:
        return _execute(reference, measurements, x)

    for a, b in zip(ours(X0), theirs(X0), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)
    # The state's phase depends on x[1] through GlobalPhase, so this checks it's traced too.
    jac_ours = jax.jacrev(lambda x: jnp.angle(ours(x)[1][0]))(X0)
    jac_theirs = jax.jacrev(lambda x: jnp.angle(theirs(x)[1][0]))(X0)
    np.testing.assert_allclose(jac_ours, jac_theirs, atol=1e-10)


def test_trainable_observable_coefficients_are_differentiated(device: Any, reference: Any) -> None:
    def energy(dev: Any, c: Any) -> Any:
        hamiltonian = c[0] * qml.PauliZ("a") + c[1] * qml.PauliX(1) @ qml.PauliY("c")
        return _execute(dev, [qml.expval(hamiltonian)], X0)

    c = jnp.array([0.7, -0.4])
    np.testing.assert_allclose(
        jax.grad(lambda c: energy(device, c))(c),
        jax.grad(lambda c: energy(reference, c))(c),
        atol=1e-10,
    )


def test_second_derivatives(device: Any, reference: Any) -> None:
    def ours(x: Any) -> Any:
        return _execute(device, [qml.expval(qml.PauliZ("c"))], x)

    def theirs(x: Any) -> Any:
        return _execute(reference, [qml.expval(qml.PauliZ("c"))], x)

    np.testing.assert_allclose(jax.hessian(ours)(X0), jax.hessian(theirs)(X0), atol=1e-10)


def test_backprop_values_equal_the_ordinary_execution(device: Any) -> None:
    measurements = [qml.expval(qml.PauliZ("a")), qml.probs(wires=[1, "c"])]
    traced = _execute(device, measurements, X0)
    ordinary = _execute(device, measurements, np.asarray(X0), config=None)
    for a, b in zip(traced, ordinary, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_complex64_keeps_its_precision() -> None:
    device = KetjetQubit(wires=WIRES, backend="jax", c_dtype="complex64")
    x = X0.astype(jnp.float32)
    state = _execute(device, [qml.state()], x)
    grad = jax.grad(lambda v: _execute(device, [qml.expval(qml.PauliZ("a"))], v))(x)
    assert state.dtype == jnp.complex64
    assert grad.dtype == jnp.float32


def test_a_circuit_without_parameters_or_operations(device: Any) -> None:
    empty = device.execute([QuantumScript([], [qml.probs(wires=["a"])])], BACKPROP)[0]
    np.testing.assert_allclose(empty, [1.0, 0.0])
    fixed = QuantumScript([qml.Hadamard(1)], [qml.expval(qml.PauliX(1))])
    assert float(device.execute([fixed], BACKPROP)[0]) == pytest.approx(1.0, abs=1e-12)


def test_one_structure_compiles_once_whatever_its_values(device: Any) -> None:
    jax_backend.clear_compile_cache()
    jax_backend._TRACES.clear()
    grad = jax.grad(lambda x: _execute(device, [qml.expval(qml.PauliZ(1))], x))
    for shift in np.linspace(-1, 1, 5):
        grad(X0 + shift)
        _execute(device, [qml.expval(qml.PauliZ(1))], X0 + shift)
    assert sum(jax_backend._TRACES.values()) == 1
