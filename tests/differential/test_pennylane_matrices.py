"""Differential tests: every ketjet gate matrix equals PennyLane's, including wire order (QS-204).

PennyLane is an independent implementation, so agreement here proves the gate definitions and the
big-endian wire convention (ADR-0001 D2) without trusting our own code. Skipped where PennyLane
isn't installed (Python 3.15 until PennyLane supports it).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from ketjet import operations as ops
from ketjet.gates import GATES
from ketjet.operations import Operation
from tests.tolerances import assert_close

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

rng = np.random.default_rng(2026)
a, b, c = (float(x) for x in rng.uniform(-np.pi, np.pi, 3))

# (id, ketjet operation, equivalent PennyLane operation), compared in each op's own wire order.
CASES: list[tuple[str, Operation, Callable[[], Any]]] = [
    ("I", ops.I(0), lambda: qml.Identity(0)),
    ("X", ops.X(0), lambda: qml.PauliX(0)),
    ("Y", ops.Y(0), lambda: qml.PauliY(0)),
    ("Z", ops.Z(0), lambda: qml.PauliZ(0)),
    ("H", ops.H(0), lambda: qml.Hadamard(0)),
    ("S", ops.S(0), lambda: qml.S(0)),
    ("Sdg", ops.Sdg(0), lambda: qml.adjoint(qml.S(0))),
    ("T", ops.T(0), lambda: qml.T(0)),
    ("Tdg", ops.Tdg(0), lambda: qml.adjoint(qml.T(0))),
    ("SX", ops.SX(0), lambda: qml.SX(0)),
    ("SXdg", Operation("SXdg", (0,)), lambda: qml.adjoint(qml.SX(0))),
    ("RX", ops.RX(a, 0), lambda: qml.RX(a, 0)),
    ("RY", ops.RY(a, 0), lambda: qml.RY(a, 0)),
    ("RZ", ops.RZ(a, 0), lambda: qml.RZ(a, 0)),
    ("PhaseShift", ops.PhaseShift(a, 0), lambda: qml.PhaseShift(a, 0)),
    ("Rot", ops.Rot(a, b, c, 0), lambda: qml.Rot(a, b, c, 0)),
    ("U3", ops.U3(a, b, c, 0), lambda: qml.U3(a, b, c, 0)),
    ("SWAP", ops.SWAP(0, 1), lambda: qml.SWAP([0, 1])),
    ("ISWAP", ops.ISWAP(0, 1), lambda: qml.ISWAP([0, 1])),
    ("ISWAPdg", Operation("ISWAPdg", (0, 1)), lambda: qml.adjoint(qml.ISWAP([0, 1]))),
    ("CNOT", ops.CNOT(0, 1), lambda: qml.CNOT([0, 1])),
    ("CNOT reversed", ops.CNOT(1, 0), lambda: qml.CNOT([1, 0])),
    ("CY", ops.CY(0, 1), lambda: qml.CY([0, 1])),
    ("CZ", ops.CZ(0, 1), lambda: qml.CZ([0, 1])),
    ("CRX", ops.CRX(a, 0, 1), lambda: qml.CRX(a, [0, 1])),
    ("CRY", ops.CRY(a, 0, 1), lambda: qml.CRY(a, [0, 1])),
    ("CRZ", ops.CRZ(a, 0, 1), lambda: qml.CRZ(a, [0, 1])),
    ("Toffoli", ops.Toffoli(0, 1, 2), lambda: qml.Toffoli([0, 1, 2])),
    ("CSWAP", ops.CSWAP(0, 1, 2), lambda: qml.CSWAP([0, 1, 2])),
    ("MultiControlledX", ops.MultiControlledX([0, 1, 2], 3, [True, False, True]),
     lambda: qml.MultiControlledX(wires=[0, 1, 2, 3], control_values=[1, 0, 1])),
    ("ctrl(H)", ops.ctrl(ops.H(1), [0]), lambda: qml.ctrl(qml.Hadamard(1), control=0)),
    ("ctrl(Rot) off-trigger", ops.ctrl(ops.Rot(a, b, c, 2), [0, 1], [False, True]),
     lambda: qml.ctrl(qml.Rot(a, b, c, 2), control=[0, 1], control_values=[0, 1])),
]  # fmt: skip


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize(("name", "op", "pl_op"), CASES, ids=[case[0] for case in CASES])
def test_matrix_matches_pennylane(
    name: str, op: Operation, pl_op: Callable[[], Any], dtype: Any
) -> None:
    pl = pl_op()
    expected = qml.matrix(pl, wire_order=list(op.all_wires))
    assert_close(op.matrix(dtype), expected, dtype)


def test_every_base_gate_is_covered() -> None:
    covered = {case[1].gate for case in CASES}
    assert covered == set(GATES)
