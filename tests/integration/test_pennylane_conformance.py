"""Regressions found by PennyLane's own device suite, ``pl-device-test`` (QS-505).

Each test pins a bug the suite found, so the main test matrix (every OS and Python version)
guards it, not only the single conformance job.
"""

from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

from pennylane import numpy as pnp  # noqa: E402

from ketjet.pennylane import _to_eigenbasis  # noqa: E402

H = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938]])


def _prepare() -> None:
    """A generic, asymmetric 3-qubit state, so a basis mistake can't cancel out."""
    qml.RX(0.4, 0)
    qml.RY(-0.7, 1)
    qml.CNOT([0, 1])
    qml.RZ(0.3, 1)
    qml.RY(1.1, 2)
    qml.CRX(0.9, [1, 2])


def _both(qfunc: Any, wires: int = 3, **kwargs: Any) -> tuple[Any, Any]:
    ours = qml.QNode(qfunc, qml.device("ketjet.qubit", wires=wires, **kwargs))()
    theirs = qml.QNode(qfunc, qml.device("default.qubit", wires=wires))()
    return ours, theirs


# --------------------------------------------------------------------------- eigenbasis rotation


def test_measurements_with_non_diagonal_observables_use_their_eigenbasis() -> None:
    """Bug: PennyLane's process_state was given the unrotated state for probs(op=...)."""

    def qfunc() -> Any:
        _prepare()
        return (
            qml.probs(op=qml.PauliX(1) @ qml.PauliY(2)),
            qml.probs(op=qml.PauliY(0) @ qml.Hadamard(1)),
            qml.expval(qml.s_prod(1j, qml.PauliX(0))),  # not a ketjet observable: falls back
        )

    ours, theirs = _both(qfunc)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_hadamard_gradient_of_probabilities() -> None:
    """Bug: hadamard_grad's probs(Z @ Y) gave all-zero gradients."""

    def build(name: str) -> Any:
        @qml.qnode(  # type: ignore[untyped-decorator]
            qml.device(name, wires=3), diff_method="hadamard", gradient_kwargs={"aux_wire": 2}
        )
        def run(x: Any, y: Any) -> Any:
            qml.RX(x, 0)
            qml.RY(y, 1)
            qml.CNOT([0, 1])
            return qml.probs(wires=[1])

        return run

    x, y = pnp.array(0.543, requires_grad=True), pnp.array(-0.654, requires_grad=True)
    ours = qml.jacobian(build("ketjet.qubit"))(x, y)
    theirs = qml.jacobian(build("default.qubit"))(x, y)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)
    assert np.abs(ours[0]).max() > 0.1


def test_sampled_hermitian_and_projector_observables() -> None:
    """Bug: shots measured Hermitian and Projector in the computational basis."""
    shots = 20_000

    def measurements() -> Any:
        _prepare()
        return (
            qml.expval(qml.Hermitian(H, wires=0)),
            qml.expval(qml.Projector([1, 0], wires=[1, 2])),
            qml.var(qml.Hermitian(H, wires=0) @ qml.PauliZ(2)),
        )

    exact = _both(measurements)[1]

    @qml.set_shots(shots)  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("ketjet.qubit", wires=3, seed=4))  # type: ignore[untyped-decorator]
    def sampled() -> Any:
        return measurements()

    estimates = sampled()
    eigvals = np.linalg.eigvalsh(H)
    spread = [eigvals.max() - eigvals.min(), 1.0, (eigvals.max() - eigvals.min()) ** 2 * 4]
    for estimate, value, width in zip(estimates, exact, spread, strict=True):
        assert abs(estimate - value) <= 4 * width / np.sqrt(shots)

    @qml.set_shots(500)  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("ketjet.qubit", wires=3, seed=5))  # type: ignore[untyped-decorator]
    def samples() -> Any:
        _prepare()
        return qml.sample(qml.Hermitian(H, wires=0))

    values = np.unique(np.round(samples(), 10))
    assert set(values) <= set(np.round(eigvals, 10))
    assert len(values) == 2  # both eigenvalues occur


# --------------------------------------------------------------------------- matrix fallback


@pytest.mark.parametrize(
    "make",
    [
        lambda: qml.DoubleExcitationPlus(0.3, wires=[0, 1, 2, 3]),
        lambda: qml.DoubleExcitationMinus(-0.8, wires=[0, 1, 2, 3]),
        lambda: qml.BlockEncode(np.array([[0.1, 0.2], [0.3, 0.4]]), wires=[1, 3]),
        lambda: qml.QSVT(
            qml.BlockEncode(np.array([[0.1]]), wires=[0, 1]),
            [qml.PCPhase(phi, dim=1, wires=[0, 1]) for phi in (0.2, 0.5, -0.3)],
        ),  # BlockEncode appears only mid-decomposition
    ],
    ids=["DoubleExcitationPlus", "DoubleExcitationMinus", "BlockEncode", "QSVT"],
)
def test_operations_without_a_decomposition_run_from_their_matrix(make: Any) -> None:
    def qfunc() -> Any:
        for w in range(4):
            qml.RY(0.3 + 0.4 * w, w)
        qml.apply(make())
        return qml.state()

    ours, theirs = _both(qfunc, wires=4)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_trainable_operations_without_a_decomposition_raise(method: str) -> None:
    """Their matrix decomposition can't be differentiated: raise, never a silent zero."""

    @qml.qnode(qml.device("ketjet.qubit", wires=4), diff_method=method)  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        qml.DoubleExcitationPlus(x, wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliX(0) @ qml.PauliY(2))

    with pytest.raises(
        qml.exceptions.DeviceError, match=r"DoubleExcitationPlus has no decomposition.*default"
    ):
        qml.grad(run)(pnp.array(0.37, requires_grad=True))


def test_one_observable_measured_twice_is_rotated_once() -> None:
    """A rotation shared by several measurements is applied once, not once per measurement.

    Through a QNode, split_non_commuting gives each Hermitian measurement its own tape, so this
    guards the helper directly, in case PennyLane's grouping ever changes.
    """
    rng = np.random.default_rng(7)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    obs = qml.Hermitian(H, wires="a")
    wires = qml.wires.Wires(["a", "b"])
    once = _to_eigenbasis(state, [qml.expval(obs)], wires)
    shared = _to_eigenbasis(state, [qml.expval(obs), qml.var(obs), qml.sample(obs)], wires)
    np.testing.assert_allclose(shared, once, atol=1e-12)
    # The rotation diagonalizes H: outcome probabilities equal those from H's eigenvectors.
    _, vectors = np.linalg.eigh(H)
    expected = np.abs(np.kron(vectors.conj().T, np.eye(2)) @ state) ** 2
    np.testing.assert_allclose(np.abs(once) ** 2, expected, atol=1e-12)
