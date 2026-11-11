"""Unsupported operations decompose into ketjet's gates; supported ones pass untouched (QS-503).

ADR-0005 D2 and D3: PennyLane's ``decompose`` stops at the supported set. Everything outside it
is rewritten by PennyLane's own rules, and the results must still match ``default.qubit``.
"""

from typing import Any

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

from ketjet.pennylane import KetjetQubit, to_ketjet  # noqa: E402
from tests.pennylane_support import random_circuit, to_pennylane  # noqa: E402


def _preprocess(ops: list[Any], wires: int = 4) -> list[Any]:
    """The operations the device receives after its preprocessing pipeline."""
    tape = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])
    (out,), _ = KetjetQubit(wires=wires).preprocess_transforms()((tape,))
    return list(out.operations)


def _random_unitary(dim: int, rng: np.random.Generator) -> Any:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


# --------------------------------------------------------------------------- supported set (D3)

SUPPORTED: list[Any] = [
    qml.Identity(0),
    qml.PauliX(0),
    qml.PauliY(1),
    qml.PauliZ(2),
    qml.Hadamard(0),
    qml.S(1),
    qml.T(2),
    qml.SX(3),
    qml.RX(0.1, 0),
    qml.RY(0.2, 1),
    qml.RZ(0.3, 2),
    qml.PhaseShift(0.4, 3),
    qml.Rot(0.1, 0.2, 0.3, 0),
    qml.U3(0.4, 0.5, 0.6, 1),
    qml.SWAP([0, 1]),
    qml.ISWAP([2, 3]),
    qml.adjoint(qml.S(0)),
    qml.adjoint(qml.T(1)),
    qml.adjoint(qml.SX(2)),
    qml.adjoint(qml.ISWAP([0, 3])),
    qml.CNOT([0, 1]),
    qml.CY([1, 2]),
    qml.CZ([2, 3]),
    qml.CRX(0.1, [0, 2]),
    qml.CRY(0.2, [1, 3]),
    qml.CRZ(0.3, [3, 0]),
    qml.CRot(0.1, 0.2, 0.3, [0, 1]),
    qml.ControlledPhaseShift(0.5, [1, 2]),
    qml.Toffoli([0, 1, 2]),
    qml.CSWAP([0, 1, 2]),
    qml.MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 0, 1]),
    qml.ctrl(qml.U3(0.1, 0.2, 0.3, 3), control=[0, 1], control_values=[0, 1]),
    qml.ctrl(qml.adjoint(qml.ISWAP([2, 3])), control=0),
    qml.GlobalPhase(0.3),
]


def test_supported_operations_reach_the_device_untouched() -> None:
    out = _preprocess(SUPPORTED)
    assert len(out) == len(SUPPORTED)
    for before, after in zip(SUPPORTED, out, strict=True):
        assert qml.equal(before, after), (before, after)


@pytest.mark.parametrize("seed", range(5))
def test_every_ketjet_gate_with_controls_is_supported(seed: int) -> None:
    """Every ketjet gate, with random controls and control values, needs no decomposition."""
    circuit = [to_pennylane(op) for op in random_circuit(4, 40, np.random.default_rng(seed))]
    out = _preprocess(circuit)
    assert len(out) == len(circuit)
    for before, after in zip(circuit, out, strict=True):
        assert qml.equal(before, after)


# --------------------------------------------------------------------------- decomposed ops


def _unsupported(rng: np.random.Generator) -> dict[str, Any]:
    u1, u2 = _random_unitary(2, rng), _random_unitary(4, rng)
    amplitudes = rng.normal(size=8) + 1j * rng.normal(size=8)
    return {
        "adjoint RX": lambda: qml.adjoint(qml.RX(0.7, 0)),
        "adjoint Rot": lambda: qml.adjoint(qml.Rot(0.1, 0.2, 0.3, 1)),
        "QubitUnitary 1": lambda: qml.QubitUnitary(u1, wires=2),
        "QubitUnitary 2": lambda: qml.QubitUnitary(u2, wires=[3, 0]),
        "ControlledQubitUnitary": lambda: qml.ControlledQubitUnitary(u1, wires=[0, 1, 3]),
        "StatePrep": lambda: qml.StatePrep(amplitudes / np.linalg.norm(amplitudes), [1, 2, 3]),
        "BasisState": lambda: qml.BasisState(np.array([1, 0, 1]), [0, 2, 3]),
        "Identity on 3 wires": lambda: qml.Identity([0, 1, 2]),
        "controlled multi-wire Identity": lambda: qml.ctrl(qml.Identity([1, 2]), 0),
        "pow S 3": lambda: qml.pow(qml.S(1), 3),
        "IsingXX": lambda: qml.IsingXX(0.3, [0, 1]),
        "IsingYY": lambda: qml.IsingYY(0.4, [1, 2]),
        "IsingZZ": lambda: qml.IsingZZ(0.5, [2, 3]),
        "MultiRZ": lambda: qml.MultiRZ(0.6, [0, 1, 3]),
        "PauliRot": lambda: qml.PauliRot(0.7, "XYZ", [0, 2, 3]),
        "SingleExcitation": lambda: qml.SingleExcitation(0.8, [0, 1]),
        "DoubleExcitation": lambda: qml.DoubleExcitation(0.9, [0, 1, 2, 3]),
        "QFT": lambda: qml.QFT(wires=[0, 1, 2, 3]),
        "StronglyEntanglingLayers": lambda: qml.StronglyEntanglingLayers(
            rng.uniform(-np.pi, np.pi, (2, 4, 3)), wires=range(4)
        ),
        "BasicEntanglerLayers": lambda: qml.BasicEntanglerLayers(
            rng.uniform(-np.pi, np.pi, (2, 4)), wires=range(4)
        ),
        "AmplitudeEmbedding": lambda: qml.AmplitudeEmbedding(
            np.arange(1.0, 9.0), wires=[0, 1, 2], normalize=True
        ),
        "controlled GlobalPhase": lambda: qml.ctrl(qml.GlobalPhase(0.4, wires=1), 0),
    }


UNSUPPORTED = sorted(_unsupported(np.random.default_rng(0)))


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_operations_decompose_and_match_default_qubit(name: str) -> None:
    rng = np.random.default_rng(len(name))
    make = _unsupported(rng)[name]
    op = make()

    def qfunc() -> Any:
        for w in range(4):
            qml.RY(0.3 + w, w)  # a non-trivial input state
        qml.apply(op)
        return qml.state()

    out = _preprocess([op])
    assert not any(qml.equal(op, o) for o in out)  # really decomposed
    index = {w: w for w in range(4)}
    for o in out:  # every result is something the device runs natively
        if not isinstance(o, qml.GlobalPhase):
            to_ketjet(o, index)
    ours = qml.QNode(qfunc, qml.device("ketjet.qubit", wires=4))()
    theirs = qml.QNode(qfunc, qml.device("default.qubit", wires=4))()
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_nested_controls_are_flattened_or_decomposed() -> None:
    nested = qml.ops.Controlled(qml.ops.Controlled(qml.RX(0.3, 2), 1), 0)

    def qfunc() -> Any:
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.apply(nested)
        return qml.state()

    ours = qml.QNode(qfunc, qml.device("ketjet.qubit", wires=3))()
    theirs = qml.QNode(qfunc, qml.device("default.qubit", wires=3))()
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


# --------------------------------------------------------------------------- errors


class _NoDecomposition(qml.operation.Operation):  # type: ignore[misc,name-defined]
    """An operation PennyLane can't decompose."""

    num_wires = 1


def test_operations_without_a_decomposition_raise_a_clear_error() -> None:
    @qml.qnode(qml.device("ketjet.qubit", wires=1))  # type: ignore[untyped-decorator]
    def run() -> Any:
        _NoDecomposition(wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(
        qml.exceptions.DeviceError,
        match=r"_NoDecomposition.*not supported with ketjet\.qubit.*does not provide a decomp",
    ):
        run()
