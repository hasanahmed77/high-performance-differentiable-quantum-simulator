"""Shared helpers for differential tests against PennyLane.

PennyLane objects are built from PennyLane's own native gates and observables (never from our
matrices), so comparisons are independent. Import this only after
``pytest.importorskip("pennylane")``.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pennylane as qml

from ketjet.gates import GATES
from ketjet.observables import Hermitian, Observable, Pauli, Prod, Sum
from ketjet.operations import Operation

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
_PL_PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def to_pennylane(op: Operation) -> Any:
    """Build the equivalent PennyLane operation (queued if called inside a QNode)."""
    base = _PL_BASE[op.gate](op.wires, op.params)
    if not op.controls:
        return base
    return qml.ctrl(base, control=list(op.controls), control_values=list(op.control_values))


def to_pennylane_observable(obs: Observable) -> Any:
    """Build the equivalent PennyLane observable."""
    if isinstance(obs, Pauli):
        if not obs.word:
            return qml.Identity(0)
        factors = [_PL_PAULI[letter](w) for w, letter in obs.word]
        return factors[0] if len(factors) == 1 else qml.prod(*factors)
    if isinstance(obs, Hermitian):
        return qml.Hermitian(np.asarray(obs.base_matrix), wires=list(obs.target_wires))
    if isinstance(obs, Prod):
        return qml.prod(*(to_pennylane_observable(f) for f in obs.factors))
    if isinstance(obs, Sum):
        return qml.sum(*(c * to_pennylane_observable(t) for c, t in obs.terms))
    raise TypeError(type(obs).__name__)  # pragma: no cover


def run_default_qubit(
    circuit: Sequence[Operation], n_wires: int, measurement: Callable[[], Any]
) -> Any:
    """Run ``circuit`` on ``default.qubit`` and return ``measurement()`` (e.g. ``qml.state``)."""
    device = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(device)  # type: ignore[untyped-decorator]  # PennyLane ships no type hints
    def run() -> Any:
        for op in circuit:
            # Constructing a PennyLane op inside a QNode queues it; don't also call qml.apply.
            to_pennylane(op)
        return measurement()

    return run()


def random_circuit(n_wires: int, depth: int, rng: np.random.Generator) -> list[Operation]:
    """Random operations over every registry gate, with random controls and control values."""
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
