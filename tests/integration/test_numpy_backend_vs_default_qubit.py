"""End-to-end: circuits run through the NumPy backend equal PennyLane's default.qubit (QS-209).

The PennyLane circuit is built from PennyLane's own native gates (not from our matrices), so this
checks gate definitions, control handling, wire order and the kernel together against an
independent simulator.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from ketjet.gates import GATES
from ketjet.operations import Operation
from tests.tolerances import assert_close

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

_PL_BASE: dict[str, Callable[..., Any]] = {
    "I": lambda w, p: qml.Identity(w[0]),
    "X": lambda w, p: qml.PauliX(w[0]),
    "Y": lambda w, p: qml.PauliY(w[0]),
    "Z": lambda w, p: qml.PauliZ(w[0]),
    "H": lambda w, p: qml.Hadamard(w[0]),
    "S": lambda w, p: qml.S(w[0]),
    "Sdg": lambda w, p: qml.adjoint(qml.S(w[0])),
    "T": lambda w, p: qml.T(w[0]),
    "Tdg": lambda w, p: qml.adjoint(qml.T(w[0])),
    "SX": lambda w, p: qml.SX(w[0]),
    "SXdg": lambda w, p: qml.adjoint(qml.SX(w[0])),
    "RX": lambda w, p: qml.RX(p[0], w[0]),
    "RY": lambda w, p: qml.RY(p[0], w[0]),
    "RZ": lambda w, p: qml.RZ(p[0], w[0]),
    "PhaseShift": lambda w, p: qml.PhaseShift(p[0], w[0]),
    "Rot": lambda w, p: qml.Rot(*p, w[0]),
    "U3": lambda w, p: qml.U3(*p, w[0]),
    "SWAP": lambda w, p: qml.SWAP(list(w)),
    "ISWAP": lambda w, p: qml.ISWAP(list(w)),
    "ISWAPdg": lambda w, p: qml.adjoint(qml.ISWAP(list(w))),
}


def to_pennylane(op: Operation) -> Any:
    base = _PL_BASE[op.gate](op.wires, op.params)
    if not op.controls:
        return base
    return qml.ctrl(base, control=list(op.controls), control_values=list(op.control_values))


def default_qubit_state(circuit: list[Operation], n_wires: int) -> Any:
    device = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(device)  # type: ignore[untyped-decorator]  # PennyLane ships no type hints
    def run() -> Any:
        for op in circuit:
            # Constructing a PennyLane op inside a QNode queues it; don't also call qml.apply.
            to_pennylane(op)
        return qml.state()

    return run()


def random_circuit(n_wires: int, depth: int, rng: np.random.Generator) -> list[Operation]:
    names = sorted(GATES)
    circuit = []
    for _ in range(depth):
        spec = GATES[names[rng.integers(len(names))]]
        n_ctrl = int(rng.integers(0, min(2, n_wires - spec.num_wires) + 1))
        wires = [int(w) for w in rng.permutation(n_wires)[: spec.num_wires + n_ctrl]]
        params = tuple(float(x) for x in rng.uniform(-np.pi, np.pi, spec.num_params))
        values = tuple(bool(v) for v in rng.integers(0, 2, n_ctrl))
        circuit.append(
            Operation(spec.name, tuple(wires[n_ctrl:]), params, tuple(wires[:n_ctrl]), values)
        )
    return circuit


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_bell_state_matches_default_qubit(dtype: Any) -> None:
    circuit = [ops.H(0), ops.CNOT(0, 1)]
    backend = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(2, dtype), circuit)
    assert_close(backend.to_numpy(state), default_qubit_state(circuit, 2), dtype)
    assert_close(backend.to_numpy(state), np.array([1, 0, 0, 1]) / np.sqrt(2), dtype)


@pytest.mark.parametrize("n_wires", [3, 5, 8])
def test_ghz_state_matches_default_qubit(n_wires: int) -> None:
    circuit = [ops.H(0)] + [ops.CNOT(i, i + 1) for i in range(n_wires - 1)]
    backend = NumPyBackend()
    state = backend.to_numpy(backend.apply_operations(backend.zero_state(n_wires), circuit))
    expected = np.zeros(2**n_wires)
    expected[0] = expected[-1] = 1 / np.sqrt(2)
    assert_close(state, expected, np.complex128)
    assert_close(state, default_qubit_state(circuit, n_wires), np.complex128)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("seed", range(25))
def test_random_circuits_match_default_qubit(seed: int, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 7))
    circuit = random_circuit(n_wires, depth=30, rng=rng)
    backend = NumPyBackend()
    state = backend.apply_operations(backend.zero_state(n_wires, dtype), circuit)
    assert state.dtype == np.dtype(dtype)
    assert_close(backend.to_numpy(state), default_qubit_state(circuit, n_wires), dtype)
