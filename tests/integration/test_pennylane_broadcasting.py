"""PennyLane parameter broadcasting on ketjet.qubit (QS-506, ADR-0007 D3).

On the JAX backend a broadcast tape runs as one ``jax.vmap``-ed execution; everywhere else it is
split into one tape per batch entry, as before. These tests work at the device level, without
PennyLane's JAX interface, so they run on every JAX version; the QNode tests that need the
interface are in ``test_pennylane_jax_interface.py``.
"""

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
from pennylane.math import Interface  # noqa: E402
from pennylane.tape import QuantumScript  # noqa: E402

from ketjet.backends import jax_backend  # noqa: E402
from ketjet.pennylane import KetjetQubit  # noqa: E402

BACKPROP = ExecutionConfig(gradient_method="backprop", interface=Interface.JAX)
WIRES = ["a", 1, "c"]
BATCH = 5
THETAS = np.linspace(-1.3, 2.1, BATCH)


def _ops(theta: Any, x: Any = 0.4) -> list[Any]:
    """Broadcast and scalar parameters, controls, a decomposed gate and a global phase."""
    return [
        qml.RX(theta, "a"),
        qml.RY(-0.7, 1),
        qml.CNOT(["a", 1]),
        qml.CRZ(x, [1, "c"]),
        qml.Rot(theta, 0.4, x, "c"),
        qml.IsingXX(theta, wires=["a", "c"]),
        qml.ctrl(qml.RY(theta, "a"), control=["c"], control_values=[0]),
        qml.GlobalPhase(theta),
    ]


MEASUREMENTS = {
    "expval": [qml.expval(0.5 * qml.PauliZ("a") @ qml.PauliX("c") - 1.1 * qml.PauliY(1))],
    "several": [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliX("a")), qml.probs(wires=["c", "a"])],
    "state": [qml.state()],
}


def _run(device: Any, tape: QuantumScript, **kwargs: Any) -> Any:
    return qml.execute([tape], device, **kwargs)[0]


def _reference(tape: QuantumScript) -> Any:
    return _run(qml.device("default.qubit", wires=WIRES), tape, diff_method=None)


def _tapes_after_preprocessing(device: Any, tape: QuantumScript, config: Any) -> int:
    config = device.setup_execution_config(config, tape)
    tapes, _ = device.preprocess_transforms(config)([tape])
    return len(tapes)


# --------------------------------------------------------------------------- values


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("kind", MEASUREMENTS)
def test_broadcast_results_equal_default_qubit(backend: str, kind: str) -> None:
    tape = QuantumScript(_ops(THETAS), MEASUREMENTS[kind])
    ours = _run(KetjetQubit(wires=WIRES, backend=backend), tape, diff_method=None)
    expected = _reference(tape)

    if len(MEASUREMENTS[kind]) == 1:
        ours, expected = (ours,), (expected,)
    assert len(ours) == len(expected)
    for got, want in zip(ours, expected, strict=True):
        assert isinstance(got, np.ndarray)
        assert np.shape(got) == np.shape(want)
        np.testing.assert_allclose(got, want, atol=1e-10)


def test_complex64_broadcasts_too() -> None:
    tape = QuantumScript(_ops(THETAS), MEASUREMENTS["expval"])
    device = KetjetQubit(wires=WIRES, backend="jax", c_dtype="complex64")
    np.testing.assert_allclose(_run(device, tape, diff_method=None), _reference(tape), atol=1e-5)


def test_several_broadcast_parameters_and_a_batch_of_one() -> None:
    def tape(thetas: Any, xs: Any) -> QuantumScript:
        return QuantumScript(_ops(thetas, xs), MEASUREMENTS["expval"])

    device = KetjetQubit(wires=WIRES, backend="jax")
    xs = np.linspace(0.2, -0.9, BATCH)
    np.testing.assert_allclose(
        _run(device, tape(THETAS, xs), diff_method=None), _reference(tape(THETAS, xs)), atol=1e-10
    )
    one = tape(THETAS[:1], 0.4)
    assert one.batch_size == 1
    np.testing.assert_allclose(_run(device, one, diff_method=None), _reference(one), atol=1e-10)


# --------------------------------------------------------------------------- one execution


def test_the_jax_backend_runs_a_broadcast_tape_as_one_execution() -> None:
    tape = QuantumScript(_ops(THETAS), MEASUREMENTS["expval"])
    jax_backend._TRACES.clear()
    jax_backend.clear_compile_cache()
    device = KetjetQubit(wires=WIRES, backend="jax")

    with qml.Tracker(device) as tracker:
        _run(device, tape, diff_method=None)
        _run(device, QuantumScript(_ops(THETAS + 0.1), MEASUREMENTS["expval"]), diff_method=None)

    # PennyLane counts the circuits hardware would run, and the simulations that ran them.
    assert tracker.totals["executions"] == 2 * BATCH
    assert tracker.totals["simulations"] == 2  # one per broadcast tape, not one per entry
    assert sum(jax_backend._TRACES.values()) == 1  # and one compilation for both

    numpy_device = KetjetQubit(wires=WIRES, backend="numpy")
    with qml.Tracker(numpy_device) as tracker:
        _run(numpy_device, tape, diff_method=None)
    assert tracker.totals["simulations"] == BATCH  # the NumPy backend still splits the tape


@pytest.mark.parametrize(
    ("backend", "config", "shots", "expected"),
    [
        ("jax", ExecutionConfig(), None, 1),
        ("jax", BACKPROP, None, 1),
        ("jax", ExecutionConfig(gradient_method="adjoint"), None, BATCH),
        ("jax", ExecutionConfig(gradient_method="parameter-shift"), None, BATCH),
        ("jax", ExecutionConfig(), 100, BATCH),
        ("numpy", ExecutionConfig(), None, BATCH),
        ("numpy", ExecutionConfig(gradient_method="adjoint"), None, BATCH),
    ],
)
def test_which_broadcast_tapes_stay_whole(
    backend: str, config: Any, shots: int | None, expected: int
) -> None:
    """Only exact tapes on the JAX backend, executed or under backprop, stay whole (D3)."""
    tape = QuantumScript(_ops(THETAS), MEASUREMENTS["expval"], shots=shots)
    assert _tapes_after_preprocessing(KetjetQubit(backend=backend), tape, config) == expected


def test_sampled_broadcast_tapes_are_split_and_sampled() -> None:
    tape = QuantumScript(_ops(THETAS), [qml.expval(qml.PauliZ(1))], shots=20_000)
    result = _run(KetjetQubit(wires=WIRES, backend="jax", seed=3), tape, diff_method=None)
    assert np.shape(result) == (BATCH,)
    exact = QuantumScript(tape.operations, tape.measurements)
    np.testing.assert_allclose(result, _reference(exact), atol=0.05)


# --------------------------------------------------------------------------- backprop


def _energies(device: Any, kind: str = "expval") -> Any:
    def energies(thetas: Any) -> Any:
        tape = QuantumScript(_ops(thetas), MEASUREMENTS[kind])
        tapes, post = device.preprocess_transforms(BACKPROP)([tape])
        return post(device.execute(tapes, BACKPROP))[0]

    return energies


def test_backprop_through_a_broadcast_tape_equals_a_loop() -> None:
    device = KetjetQubit(wires=WIRES, backend="jax")
    energies = _energies(device)
    thetas = jnp.asarray(THETAS)

    jac = jax.jacrev(energies)(thetas)
    looped = jnp.stack([jax.grad(lambda t: energies(t[None])[0])(t) for t in thetas])

    assert jac.shape == (BATCH, BATCH)
    np.testing.assert_allclose(jnp.diagonal(jac), looped, atol=1e-10)
    np.testing.assert_allclose(jac - jnp.diag(jnp.diagonal(jac)), 0, atol=1e-12)  # independent
    np.testing.assert_allclose(jax.jit(jax.jacrev(energies))(thetas), jac, atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(energies)(thetas),
        _reference(QuantumScript(_ops(THETAS), MEASUREMENTS["expval"])),
        atol=1e-10,
    )


def test_backprop_broadcasts_every_measurement_type() -> None:
    device = KetjetQubit(wires=WIRES, backend="jax")
    expval, var, probs = _energies(device, "several")(jnp.asarray(THETAS))
    assert (expval.shape, var.shape, probs.shape) == ((BATCH,), (BATCH,), (BATCH, 4))
    grad = jax.grad(lambda t: jnp.sum(_energies(device, "several")(t)[1]))(jnp.asarray(THETAS))
    assert grad.shape == (BATCH,)


def test_a_broadcast_tape_without_a_global_phase() -> None:
    tape = QuantumScript([qml.RX(THETAS, 0), qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(1))])
    result = _run(KetjetQubit(wires=2, backend="jax"), tape, diff_method=None)
    np.testing.assert_allclose(result, np.cos(THETAS), atol=1e-10)
