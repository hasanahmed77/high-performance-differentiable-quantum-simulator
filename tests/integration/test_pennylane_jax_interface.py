"""QNodes on ketjet.qubit under PennyLane's JAX interface, end to end (QS-608, ADR-0006 D8).

PennyLane 0.45's JAX interface needs JAX < 0.11 (ADR-0006 D1), so the main suite, which runs the
latest JAX, skips this module. CI's conformance job runs it on JAX 0.10.
"""

from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
jax = pytest.importorskip("jax")

from tests.pennylane_support import (  # noqa: E402
    PENNYLANE_JAX_DEPRECATIONS,
    PENNYLANE_JAX_ONLY,
    PENNYLANE_JAX_WORKS,
)

if not PENNYLANE_JAX_WORKS:
    pytest.skip(PENNYLANE_JAX_ONLY, allow_module_level=True)

import jax.numpy as jnp  # noqa: E402
from pennylane.exceptions import QuantumFunctionError  # noqa: E402

pytestmark = [
    pytest.mark.pennylane,
    # PennyLane 0.45 calls jax.core functions deprecated in JAX 0.10 (ADR-0006 D1).
    pytest.mark.filterwarnings(PENNYLANE_JAX_DEPRECATIONS),
]

METHODS = ["backprop", "adjoint", "parameter-shift"]
X0 = jnp.array([0.3, -1.2, 0.8])


def _layers(x: Any) -> None:
    """Gates ketjet runs natively, and ones PennyLane decomposes first (CRot, IsingXX, QFT)."""
    qml.RX(x[0], 0)
    qml.RY(-0.7, 1)
    qml.CRot(x[1], x[0], 0.4, wires=[0, 1])
    qml.IsingXX(x[2], wires=[1, 2])
    qml.QFT(wires=[0, 1, 2])
    qml.CNOT([2, 0])
    qml.RZ(x[1], 2)


def energy(x: Any) -> Any:
    _layers(x)
    return qml.expval(0.5 * qml.PauliZ(0) @ qml.PauliX(2) - 1.1 * qml.PauliY(1))


def _qnode(qfunc: Any, method: str, device: str = "ketjet.qubit", **kwargs: Any) -> Any:
    dev = qml.device(device, wires=3, **kwargs)
    return qml.QNode(qfunc, dev, interface="jax", diff_method=method)


@pytest.fixture(scope="module")
def expected() -> tuple[Any, Any]:
    ref = _qnode(energy, "backprop", device="default.qubit")
    return ref(X0), jax.grad(ref)(X0)


@pytest.mark.parametrize("method", METHODS)
def test_value_grad_and_jit_grad_match_default_qubit(method: str, expected: Any) -> None:
    value, grad = expected
    ours = _qnode(energy, method, backend="jax")
    np.testing.assert_allclose(ours(X0), value, atol=1e-10)
    np.testing.assert_allclose(jax.grad(ours)(X0), grad, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(ours))(X0), grad, atol=1e-10)
    np.testing.assert_allclose(jax.jit(ours)(X0), value, atol=1e-10)


def test_backprop_handles_every_measurement_type_and_jacobians() -> None:
    def circuit(x: Any) -> Any:
        _layers(x)
        return qml.expval(qml.PauliZ(1)), qml.var(qml.PauliX(0)), qml.probs(wires=[2, 0])

    ours, ref = (
        _qnode(circuit, "backprop", backend="jax"),
        _qnode(circuit, "backprop", "default.qubit"),
    )
    for a, b in zip(jax.jit(ours)(X0), ref(X0), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)
    jac_ours = jax.jit(jax.jacobian(lambda x: jnp.hstack(ours(x))))(X0)
    np.testing.assert_allclose(jac_ours, jax.jacobian(lambda x: jnp.hstack(ref(x)))(X0), atol=1e-10)


def test_backprop_gives_second_derivatives() -> None:
    ours = _qnode(energy, "backprop", backend="jax")
    ref = _qnode(energy, "backprop", device="default.qubit")
    np.testing.assert_allclose(jax.hessian(ours)(X0), jax.hessian(ref)(X0), atol=1e-10)


def test_the_jax_backends_backprop_trains_a_circuit_under_jit() -> None:
    qnode = _qnode(energy, "backprop", backend="jax")

    def descend(x: Any) -> Any:
        return x - 0.2 * jax.grad(qnode)(x)

    step = jax.jit(descend)

    x, start = X0, float(qnode(X0))
    for _ in range(30):
        x = step(x)
    assert float(qnode(x)) < start - 0.1


def test_backprop_on_the_numpy_backend_still_raises_pennylanes_error() -> None:
    with pytest.raises(QuantumFunctionError, match="does not support backprop"):
        _qnode(energy, "backprop")


def test_best_still_means_adjoint_on_the_jax_backend(expected: Any) -> None:
    ours = _qnode(energy, "best", backend="jax")
    np.testing.assert_allclose(jax.grad(ours)(X0), expected[1], atol=1e-10)
    tape = qml.workflow.construct_tape(ours)(X0)
    config = ours.device.setup_execution_config(
        qml.devices.ExecutionConfig(gradient_method="best"), tape
    )
    assert config.gradient_method == "adjoint"


# --------------------------------------------------------------------------- broadcasting (QS-506)


def broadcast_energy(thetas: Any, x: Any) -> Any:
    qml.RX(thetas, 0)
    qml.CRot(x[1], x[0], 0.4, wires=[0, 1])
    qml.IsingXX(thetas, wires=[1, 2])
    qml.QFT(wires=[0, 1, 2])
    qml.RZ(x[2], 2)
    return qml.expval(0.5 * qml.PauliZ(0) @ qml.PauliX(2) - 1.1 * qml.PauliY(1))


THETAS = jnp.linspace(-1.1, 2.3, 4)


@pytest.mark.parametrize("method", METHODS)
def test_broadcast_qnodes_match_default_qubit(method: str) -> None:
    """ADR-0007 D3: values, Jacobians and jit equal default.qubit's, for every method."""
    ours = _qnode(broadcast_energy, method, backend="jax")
    ref = _qnode(broadcast_energy, "backprop", device="default.qubit")

    np.testing.assert_allclose(ours(THETAS, X0), ref(THETAS, X0), atol=1e-10)
    # PennyLane's parameter-shift can't differentiate broadcast parameters on any device (#4462).
    for argnums in (1,) if method == "parameter-shift" else (0, 1):
        expected = jax.jacrev(ref, argnums=argnums)(THETAS, X0)
        np.testing.assert_allclose(
            jax.jacrev(ours, argnums=argnums)(THETAS, X0), expected, atol=1e-10
        )
    np.testing.assert_allclose(jax.jit(ours)(THETAS, X0), ref(THETAS, X0), atol=1e-10)


@pytest.mark.parametrize(("method", "simulations"), [("backprop", 1), ("adjoint", len(THETAS))])
def test_a_broadcast_qnode_runs_once_under_backprop(method: str, simulations: int) -> None:
    ours = _qnode(broadcast_energy, method, backend="jax")
    with qml.Tracker(ours.device) as tracker:
        ours(THETAS, X0)
    assert tracker.totals["simulations"] == simulations
