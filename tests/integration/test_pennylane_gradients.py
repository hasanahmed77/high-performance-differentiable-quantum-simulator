"""Gradients of QNodes on ``ketjet.qubit`` (QS-504, ADR-0005 D4).

Device adjoint Jacobians and VJPs must equal ``default.qubit``'s backprop, which differentiates
PennyLane's own gates by automatic differentiation and shares no code with ketjet. PennyLane's
parameter-shift and finite differences run through the device's ordinary execution.
"""

from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pytestmark = [
    pytest.mark.pennylane,
    # autograd's own complex-to-real cast inside default.qubit's backprop; not our code.
    pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning"),
]

from pennylane import numpy as pnp  # noqa: E402
from pennylane.devices import ExecutionConfig  # noqa: E402

from ketjet.backends import NumPyBackend  # noqa: E402
from ketjet.pennylane import KetjetQubit  # noqa: E402
from tests.pennylane_support import random_circuit, to_pennylane  # noqa: E402
from tests.random_circuits import symbolize  # noqa: E402
from tests.tolerances import GRADIENT_FD, assert_close  # noqa: E402

DTYPES = [np.complex128, np.complex64]


class CountingBackend(NumPyBackend):
    """Records the number of observables of every adjoint call."""

    def __init__(self) -> None:
        self.adjoint_calls: list[int] = []

    def adjoint_jacobian(self, *args: Any, **kwargs: Any) -> Any:
        self.adjoint_calls.append(len(args[2]))
        return super().adjoint_jacobian(*args, **kwargs)


def _jacobians(qfunc: Any, wires: int, params: Any, **ours: Any) -> tuple[Any, Any]:
    """Jacobian on ketjet.qubit (adjoint unless ``diff_method`` is given) and default.qubit."""
    diff_method = ours.pop("diff_method", "adjoint")
    device = qml.device("ketjet.qubit", wires=wires, **ours)
    mine = qml.QNode(qfunc, device, diff_method=diff_method)
    reference = qml.QNode(qfunc, qml.device("default.qubit", wires=wires), diff_method="backprop")

    def stacked(node: Any) -> Any:
        return lambda x: qml.math.stack(node(x))

    return qml.jacobian(stacked(mine))(params), qml.jacobian(stacked(reference))(params)


# --------------------------------------------------------------------------- agreement


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(10))
def test_adjoint_matches_default_qubit_backprop(seed: int, dtype: Any) -> None:
    """Random circuits over every ketjet gate, with shared parameters, several observables."""
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(2, 5))
    ops, values, slot_symbol = symbolize(random_circuit(n, depth=15, rng=rng), rng)
    theta = pnp.array(list(values.values()), requires_grad=True)
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    observables = [
        qml.PauliZ(0),
        qml.PauliX(0) @ qml.PauliY(1),
        0.3 * qml.PauliZ(1) - 1.2 * qml.PauliX(n - 1),
        qml.Hermitian(m + m.conj().T, wires=n - 1),
    ]

    def qfunc(x: Any) -> Any:
        slot = 0
        for op in ops:
            to_pennylane(op, [x[slot_symbol[slot + j]] for j in range(len(op.params))])
            slot += len(op.params)
        return [qml.expval(o) for o in observables]

    ours, theirs = _jacobians(qfunc, n, theta, c_dtype=dtype)
    assert_close(ours, theirs, dtype)


def _decomposed(x: Any) -> Any:
    """Gates that only reach the device through decomposition, plus CRot and a GlobalPhase."""
    qml.StronglyEntanglingLayers(qml.math.reshape(x[:6], (1, 2, 3)), wires=[0, 1])
    qml.adjoint(qml.RX(x[6], 0))  # decomposes to RX(-x): the 1:1 rule must still hold
    qml.CRot(x[7], x[8], x[6], wires=[1, 2])
    qml.IsingXX(x[8], wires=[0, 2])
    qml.DoubleExcitation(x[7], wires=[0, 1, 2, 3])
    qml.GlobalPhase(x[0])  # trainable, but no expectation value depends on it
    return [qml.expval(qml.PauliZ(0) @ qml.PauliZ(3)), qml.expval(qml.PauliX(2))]


def test_adjoint_through_decompositions_matches_backprop() -> None:
    x = pnp.array(np.linspace(-1.2, 1.4, 9), requires_grad=True)
    ours, theirs = _jacobians(_decomposed, 4, x)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_trainable_global_phase_has_zero_derivative() -> None:
    def qfunc(x: Any) -> Any:
        qml.RY(x[1], 0)
        qml.GlobalPhase(x[0])
        return [qml.expval(qml.PauliZ(0))]

    ours, theirs = _jacobians(qfunc, 1, pnp.array([0.3, 0.8], requires_grad=True))
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
    assert ours[0, 0] == 0


@pytest.mark.parametrize("method", ["parameter-shift", "finite-diff"])
def test_pennylane_gradient_transforms_run_on_the_device(method: str) -> None:
    x = pnp.array(np.linspace(-1.2, 1.4, 9), requires_grad=True)
    ours, theirs = _jacobians(_decomposed, 4, x, diff_method=method)
    atol = GRADIENT_FD.atol if method == "finite-diff" else 1e-10
    np.testing.assert_allclose(ours, theirs, atol=atol)


def test_parameter_shift_with_shots_is_reproducible_and_close() -> None:
    def build(seed: int) -> Any:
        @qml.set_shots(20_000)  # type: ignore[untyped-decorator]
        @qml.qnode(qml.device("ketjet.qubit", wires=2, seed=seed), diff_method="parameter-shift")  # type: ignore[untyped-decorator]
        def run(x: Any) -> Any:
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(1))

        return run

    x = pnp.array(0.7, requires_grad=True)
    first = qml.grad(build(1))(x)
    assert first == qml.grad(build(1))(x)
    # d cos(x) / dx = -sin(x); each of the two shifted terms has standard deviation <= 1/sqrt(N).
    assert abs(first + np.sin(0.7)) < 4 * np.sqrt(2 / 20_000)


# --------------------------------------------------------------------------- method resolution


def test_best_resolves_to_device_adjoint() -> None:
    backend = CountingBackend()

    @qml.qnode(KetjetQubit(wires=2, backend=backend), diff_method="best")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        qml.RY(x, 0)
        qml.CNOT([0, 1])
        return qml.expval(qml.PauliZ(1))

    x = pnp.array(0.4, requires_grad=True)
    np.testing.assert_allclose(qml.grad(run)(x), -np.sin(0.4), atol=1e-12)
    assert backend.adjoint_calls == [1]
    assert qml.workflow.get_best_diff_method(run)(x) == "adjoint"


def test_best_falls_back_to_parameter_shift_where_adjoint_cannot_run() -> None:
    backend = CountingBackend()

    @qml.qnode(KetjetQubit(wires=1, backend=backend), diff_method="best")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        qml.RX(x, 0)
        return qml.var(qml.PauliZ(0))

    x = pnp.array(0.4, requires_grad=True)
    np.testing.assert_allclose(qml.grad(run)(x), np.sin(0.8), atol=1e-10)  # Var = sin^2 x
    assert backend.adjoint_calls == []


@pytest.mark.parametrize(
    "measure",
    [
        lambda: qml.var(qml.PauliZ(0)),
        lambda: qml.probs(wires=[0]),
        qml.state,
        lambda: (qml.expval(qml.PauliZ(0)), qml.probs(wires=[0])),
    ],
)
def test_adjoint_rejects_measurements_other_than_expval(measure: Any) -> None:
    @qml.qnode(qml.device("ketjet.qubit", wires=1), diff_method="adjoint")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        qml.RX(x, 0)
        return measure()

    with pytest.raises(qml.exceptions.QuantumFunctionError, match="does not support adjoint"):
        run(pnp.array(0.1, requires_grad=True))


def test_adjoint_rejects_shots() -> None:
    @qml.set_shots(100)  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("ketjet.qubit", wires=1), diff_method="adjoint")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        qml.RX(x, 0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(qml.exceptions.QuantumFunctionError, match="does not support adjoint"):
        run(pnp.array(0.1, requires_grad=True))


def test_adjoint_rejects_trainable_observables() -> None:
    @qml.qnode(qml.device("ketjet.qubit", wires=1), diff_method="adjoint")  # type: ignore[untyped-decorator]
    def run(m: Any) -> Any:
        qml.RX(0.3, 0)
        return qml.expval(qml.Hermitian(m, wires=0))

    with pytest.raises(qml.exceptions.QuantumFunctionError, match="adjoint"):
        qml.grad(run)(pnp.array(np.diag([1.0, -1.0]), requires_grad=True))


def test_backprop_is_not_supported() -> None:
    """ADR-0005 D4: NumPy kernels can't be traced; backprop comes with the JAX backend."""
    with pytest.raises(qml.exceptions.QuantumFunctionError, match="does not support backprop"):
        qml.QNode(
            lambda: qml.expval(qml.PauliZ(0)),
            qml.device("ketjet.qubit", wires=1),
            diff_method="backprop",
        )


def test_only_first_order_adjoint_derivatives() -> None:
    dev = KetjetQubit(wires=1)
    assert dev.supports_derivatives()
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="adjoint"))
    assert dev.supports_derivatives(ExecutionConfig(gradient_method="device"))
    assert not dev.supports_derivatives(
        ExecutionConfig(gradient_method="adjoint", derivative_order=2)
    )
    assert not dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
    assert not dev.supports_derivatives(ExecutionConfig(gradient_method="parameter-shift"))
    tape = qml.tape.QuantumScript([qml.RX(0.1, 0)], [])
    assert not dev.supports_derivatives(ExecutionConfig(gradient_method="adjoint"), tape)


# --------------------------------------------------------------------------- VJP (ADR-0004 D6)


def test_device_vjp_is_one_backward_pass_for_many_measurements() -> None:
    backend = CountingBackend()
    weights = pnp.array([1.0, -2.0, 0.5, 3.0])

    def qfunc(x: Any) -> Any:
        qml.RY(x[0], 0)
        qml.CNOT([0, 1])
        qml.RX(x[1], 1)
        qml.CRZ(x[0], [1, 2])
        return [qml.expval(o) for o in (qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(1), qml.PauliY(2))]

    mine = qml.QNode(
        qfunc, KetjetQubit(wires=3, backend=backend), diff_method="adjoint", device_vjp=True
    )
    reference = qml.QNode(qfunc, qml.device("default.qubit", wires=3), diff_method="backprop")
    x = pnp.array([0.3, -0.9], requires_grad=True)
    ours = qml.grad(lambda a: qml.math.sum(qml.math.stack(mine(a)) * weights))(x)
    theirs = qml.grad(lambda a: qml.math.sum(qml.math.stack(reference(a)) * weights))(x)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
    assert backend.adjoint_calls == [1]  # one combined observable, not four


def _vjp_tape() -> Any:
    tape = qml.tape.QuantumScript(
        [qml.RX(0.3, 0), qml.CNOT([0, 1]), qml.RY(-0.8, 1), qml.GlobalPhase(0.2)],
        [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.expval(qml.PauliZ(1))],
    )
    tape.trainable_params = [0, 1, 2]  # RX, RY and the global phase
    return tape


def test_batched_cotangents_equal_the_contracted_jacobian() -> None:
    dev = KetjetQubit(wires=2)
    tape = _vjp_tape()
    jac = np.array(dev.compute_derivatives([tape])[0])  # (measurements, parameters)
    v = np.random.default_rng(0).normal(size=(3, 5))  # 5 cotangents at once
    (vjp,) = dev.compute_vjp([tape], [v])
    assert vjp.shape == (3, 5)
    np.testing.assert_allclose(vjp, jac.T @ v, atol=1e-12)
    (single,) = dev.compute_vjp([tape], [v[:, 0]])
    np.testing.assert_allclose(single, jac.T @ v[:, 0], atol=1e-12)


def test_zero_cotangent_needs_no_simulation() -> None:
    backend = CountingBackend()
    dev = KetjetQubit(wires=2, backend=backend)
    (vjp,) = dev.compute_vjp([_vjp_tape()], [np.zeros(3)])
    np.testing.assert_array_equal(vjp, np.zeros(3))
    assert backend.adjoint_calls == []


def test_single_measurement_vjp() -> None:
    dev = KetjetQubit(wires=1)
    tape = qml.tape.QuantumScript([qml.RX(0.4, 0)], [qml.expval(qml.PauliZ(0))])
    (vjp,) = dev.compute_vjp([tape], [2.0])
    np.testing.assert_allclose(vjp, [-2 * np.sin(0.4)], atol=1e-12)
    (batched,) = dev.compute_vjp([tape], [np.array([1.0, 3.0])])
    np.testing.assert_allclose(batched, [[-np.sin(0.4), -3 * np.sin(0.4)]], atol=1e-12)


def test_execute_and_compute_vjp() -> None:
    dev = KetjetQubit(wires=2)
    tape = _vjp_tape()
    results, vjps = dev.execute_and_compute_vjp([tape], [np.ones(3)])
    np.testing.assert_allclose(results[0], dev.execute([tape])[0], atol=1e-12)
    np.testing.assert_allclose(vjps[0], dev.compute_vjp([tape], [np.ones(3)])[0], atol=1e-12)


# --------------------------------------------------------------------------- result shapes


@pytest.mark.parametrize(("n_params", "n_measurements"), [(1, 1), (3, 1), (1, 2), (2, 3)])
def test_jacobian_shapes_match_default_qubit(n_params: int, n_measurements: int) -> None:
    ops = [qml.RX(0.1 * (k + 1), k % 2) for k in range(n_params)] + [qml.CNOT([0, 1])]
    meas = [qml.expval(qml.PauliZ(k % 2)) for k in range(n_measurements)]
    tape = qml.tape.QuantumScript(ops, meas)
    tape.trainable_params = list(range(n_params))
    config = ExecutionConfig(gradient_method="adjoint")
    (ours,) = KetjetQubit(wires=2).compute_derivatives([tape], config)
    (theirs,) = qml.device("default.qubit", wires=2).compute_derivatives([tape], config)
    assert type(ours) is type(theirs)
    assert np.shape(ours) == np.shape(theirs)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_only_a_global_phase_trainable_needs_no_backend_call() -> None:
    backend = CountingBackend()
    tape = qml.tape.QuantumScript(
        [qml.RX(0.3, 0), qml.GlobalPhase(0.2)], [qml.expval(qml.PauliZ(0))]
    )
    tape.trainable_params = [1]
    (jac,) = KetjetQubit(wires=1, backend=backend).compute_derivatives([tape])
    np.testing.assert_array_equal(jac, 0.0)
    assert backend.adjoint_calls == []


def test_trainable_observable_parameters_get_zero_derivatives() -> None:
    """Adjoint differentiates the circuit only; PennyLane warns about observable parameters."""
    matrix = pnp.array(np.diag([1.0, -1.0]), requires_grad=True)
    tape = qml.tape.QuantumScript(
        [qml.GlobalPhase(0.1), qml.RX(0.3, 0)], [qml.expval(qml.Hermitian(matrix, wires=0))]
    )
    tape.trainable_params = [0, 1, 2]  # the global phase, RX and the observable's matrix
    program = KetjetQubit(wires=1).preprocess_transforms(ExecutionConfig(gradient_method="adjoint"))
    with pytest.warns(UserWarning, match="input parameters of Hermitian"):
        program((tape,))
    (jac,) = KetjetQubit(wires=1).compute_derivatives([tape])
    np.testing.assert_allclose(np.array(jac), [0.0, -np.sin(0.3), 0.0], atol=1e-12)


def test_trainable_state_preparation_through_its_decomposition() -> None:
    """StatePrep becomes rotations; PennyLane chains the gradient through that rewrite."""

    def qfunc(x: Any) -> Any:
        qml.AmplitudeEmbedding(x, wires=[0, 1], normalize=True)
        qml.CNOT([0, 1])
        qml.RY(0.4, 1)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))]

    x = pnp.array([0.3, -1.1, 0.8, 0.5], requires_grad=True)
    ours, theirs = _jacobians(qfunc, 2, x)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)
