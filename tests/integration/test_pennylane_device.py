"""The ``ketjet.qubit`` PennyLane device (QS-502, ADR-0005).

QNodes run on ``ketjet.qubit`` and on PennyLane's ``default.qubit``, which shares no simulation
code with ketjet, and the results must agree at the standard tolerances.
"""

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

from ketjet import operations as ops  # noqa: E402
from ketjet.backends import NumPyBackend  # noqa: E402
from ketjet.observables import Hermitian, Pauli, Prod, Sum  # noqa: E402
from ketjet.operations import Operation  # noqa: E402
from ketjet.pennylane import KetjetQubit, to_ketjet, to_ketjet_observable  # noqa: E402
from tests.pennylane_support import random_circuit, to_pennylane  # noqa: E402
from tests.tolerances import assert_close  # noqa: E402

DTYPES = [np.complex128, np.complex64]
LABELS = ["a", 3, "q", "w1", 0.5]  # arbitrary wire labels (a tuple would read as several wires)


def _both(qfunc: Any, wires: Any, **device_kwargs: Any) -> tuple[Any, Any]:
    ours = qml.QNode(qfunc, qml.device("ketjet.qubit", wires=wires, **device_kwargs))()
    theirs = qml.QNode(qfunc, qml.device("default.qubit", wires=wires))()
    return ours, theirs


def _random_observables(labels: list[Any], rng: np.random.Generator) -> list[Any]:
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    w = [labels[int(i)] for i in rng.permutation(len(labels))]
    return [
        qml.PauliZ(w[0]),
        qml.PauliX(w[0]) @ qml.PauliY(w[1]),
        0.4 * qml.PauliZ(w[0]) @ qml.PauliZ(w[1]) - 1.3 * qml.PauliX(w[1]),
        qml.Hermitian(m + m.conj().T, wires=[w[1], w[0]]),
        qml.Hadamard(w[1]),  # no Pauli form: becomes a Hermitian
    ]


# --------------------------------------------------------------------------- registration


def test_entry_point_resolves_to_the_device() -> None:
    dev = qml.device("ketjet.qubit", wires=3)
    assert isinstance(dev, KetjetQubit)
    assert dev.name == "ketjet.qubit"
    assert dev.c_dtype == np.complex128


def test_import_ketjet_does_not_import_pennylane() -> None:
    code = "import sys, ketjet, ketjet.gradients; print('pennylane' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        KetjetQubit(wires=1, c_dtype="float64")


# --------------------------------------------------------------------------- differential


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", range(15))
def test_random_qnodes_match_default_qubit(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(2, 6))
    labels = LABELS[:n]
    circuit = random_circuit(n, depth=20, rng=rng)
    observables = _random_observables(labels, rng)
    subset = [labels[int(i)] for i in rng.permutation(n)[: int(rng.integers(1, n + 1))]]

    def qfunc() -> Any:
        mapped = qml.map_wires(
            lambda: [to_pennylane(op) for op in circuit], dict(enumerate(labels))
        )
        mapped()
        return (
            qml.state(),
            qml.probs(wires=subset),
            *(qml.expval(o) for o in observables),
            *(qml.var(o) for o in observables),
        )

    ours, theirs = _both(qfunc, labels, c_dtype=dtype)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        assert_close(a, b, dtype)
    assert np.asarray(ours[0]).dtype == dtype


def test_global_phase_reaches_the_state_exactly() -> None:
    def qfunc() -> Any:
        qml.Hadamard(0)
        qml.GlobalPhase(0.7)
        qml.ctrl(qml.GlobalPhase(0.3, wires=1), control=0)  # decomposed: a relative phase
        return qml.state()

    ours, theirs = _both(qfunc, 2)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_measurements_computed_by_pennylane() -> None:
    """State-based measurements outside ketjet's set use PennyLane's definitions (D5)."""

    def qfunc() -> Any:
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.RY(0.3, 1)
        return (
            qml.density_matrix([0]),
            qml.purity([0]),
            qml.vn_entropy([0]),
            qml.probs(op=qml.PauliX(1)),
            qml.expval(qml.PauliX(0) @ qml.PauliX(0)),  # overlapping product: a Hermitian
        )

    ours, theirs = _both(qfunc, 2)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_observables_ketjet_cant_represent_fall_back_to_pennylane() -> None:
    """A complex coefficient isn't a ketjet observable: PennyLane decides what it means."""

    def qfunc() -> Any:
        qml.RY(0.4, 0)
        return qml.expval(qml.s_prod(1j, qml.PauliZ(0)))

    ours, theirs = _both(qfunc, 1)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
    assert np.iscomplexobj(theirs)  # the case really is outside ketjet's real observables


def test_probs_of_all_wires_by_default() -> None:
    def qfunc() -> Any:
        qml.PauliX("y")
        return qml.probs()

    ours, theirs = _both(qfunc, ["x", "y"])
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_device_without_wires_uses_the_circuit_wires() -> None:
    def qfunc() -> Any:
        qml.PauliX("b")
        qml.CNOT(["b", "a"])
        return qml.state()

    ours, theirs = _both(qfunc, None)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_mid_circuit_measurements_are_deferred() -> None:
    def qfunc() -> Any:
        qml.Hadamard(0)
        m = qml.measure(0)
        qml.cond(m, qml.PauliX)(1)
        return qml.probs(wires=[0, 1])

    ours, theirs = _both(qfunc, None)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_custom_backend_runs_exactly_the_circuit() -> None:
    applied: list[str] = []

    class Counting(NumPyBackend):
        def apply_operation(self, state: Any, op: Operation) -> Any:
            applied.append(op.gate)
            return super().apply_operation(state, op)

    @qml.qnode(KetjetQubit(wires=2, backend=Counting()))  # type: ignore[untyped-decorator]
    def run() -> Any:
        qml.Hadamard(0)
        qml.CRX(0.2, [0, 1])
        return qml.expval(qml.PauliX(1))

    run()
    # Exact results keep the observable: no basis-change gates are added (they are with shots).
    assert applied == ["H", "RX"]


def test_tracker_counts_executions() -> None:
    dev = qml.device("ketjet.qubit", wires=1)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]
    def run() -> Any:
        return qml.expval(qml.PauliZ(0))

    with qml.Tracker(dev) as tracker:
        run()
        run()
    assert tracker.totals["executions"] == 2


# --------------------------------------------------------------------------- shots


def _prepare() -> None:
    qml.RY(1.1, 0)
    qml.CNOT([0, 1])
    qml.RX(0.4, 1)


def _shot_qnode(seed: Any, shots: Any) -> Any:
    @qml.set_shots(shots)  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("ketjet.qubit", wires=2, seed=seed))  # type: ignore[untyped-decorator]
    def run() -> Any:
        _prepare()
        return (
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliX(1)),  # doesn't commute with Z(1): split into its own tape
            qml.sample(wires=[0, 1]),
            qml.counts(wires=[0]),
        )

    return run


def test_seeded_shots_are_reproducible() -> None:
    first = _shot_qnode(7, 500)()
    second = _shot_qnode(7, 500)()
    assert first[0] == second[0]
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])
    assert first[3] == second[3]
    assert _shot_qnode(8, 500)()[0] != first[0]


def test_shot_estimates_are_within_four_sigma() -> None:
    shots = 20_000

    def analytic() -> Any:
        _prepare()
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliX(1))

    exact = _both(analytic, 2)[1]  # default.qubit's exact values
    estimates = _shot_qnode(11, shots)()
    for estimate, value in zip(estimates[:2], exact, strict=True):
        sigma = np.sqrt((1 - value**2) / shots)
        assert abs(estimate - value) <= 4 * sigma
    assert estimates[2].shape == (shots, 2)
    assert sum(estimates[3].values()) == shots


def test_shot_vectors_return_one_result_per_bin() -> None:
    result = _shot_qnode(3, [100, (50, 2), 300])()
    assert len(result) == 4
    assert [r[2].shape[0] for r in result] == [100, 50, 50, 300]


def test_shots_reject_state_measurements() -> None:
    @qml.set_shots(10)  # type: ignore[untyped-decorator]
    @qml.qnode(qml.device("ketjet.qubit", wires=1))  # type: ignore[untyped-decorator]
    def run() -> Any:
        return qml.state()

    with pytest.raises(qml.exceptions.DeviceError, match=r"ketjet\.qubit"):
        run()


# --------------------------------------------------------------------------- translation


@pytest.mark.parametrize("seed", range(5))
def test_translation_is_one_to_one_with_the_same_parameters(seed: int) -> None:
    """ADR-0005 D2: each supported PennyLane op becomes one ketjet op, parameters unchanged."""
    index = {w: w for w in range(4)}
    for op in random_circuit(4, depth=40, rng=np.random.default_rng(seed)):
        pl = to_pennylane(op)
        assert to_ketjet(pl, index) == op
        assert list(pl.parameters) == list(op.params)


def test_translation_maps_wire_labels() -> None:
    op = to_ketjet(qml.CRX(0.3, wires=["c", "t"]), {"t": 0, "c": 1})
    assert op == ops.CRX(0.3, 1, 0)


def test_unsupported_operations_are_not_translated() -> None:
    with pytest.raises(ValueError, match="QubitUnitary"):
        to_ketjet(qml.QubitUnitary(np.eye(2), wires=0), {0: 0})
    with pytest.raises(ValueError, match="should be decomposed"):
        to_ketjet(qml.adjoint(qml.RX(0.3, 0)), {0: 0})  # would need -0.3: decomposed instead


def test_observable_translation() -> None:
    index = {"a": 0, "b": 1}
    assert isinstance(to_ketjet_observable(qml.PauliX("a") @ qml.PauliZ("b"), index), Pauli)
    assert to_ketjet_observable(qml.Identity("a"), index) == Pauli(())
    assert isinstance(to_ketjet_observable(qml.Hadamard("b"), index), Hermitian)
    assert isinstance(
        to_ketjet_observable(qml.Hermitian(np.eye(2), "a") @ qml.PauliX("b"), index), Prod
    )
    assert isinstance(to_ketjet_observable(qml.PauliX("a") @ qml.PauliX("a"), index), Hermitian)
    total = to_ketjet_observable(
        qml.Hamiltonian([0.5, -1.0], [qml.PauliZ("a"), qml.PauliX("b")]), index
    )
    assert isinstance(total, Sum)
    assert [c for c, _ in total.terms] == [0.5, -1.0]
    with pytest.raises(ValueError, match="must be real"):
        to_ketjet_observable(1j * qml.PauliZ("a"), index)


def test_translated_observables_have_pennylanes_matrices() -> None:
    rng = np.random.default_rng(5)
    for obs in _random_observables(["x", "y", "z"], rng):
        ours = to_ketjet_observable(obs, {"x": 0, "y": 1, "z": 2})
        theirs = qml.matrix(obs, wire_order=["x", "y", "z"])
        np.testing.assert_allclose(ours.matrix(3), theirs, atol=1e-12)
