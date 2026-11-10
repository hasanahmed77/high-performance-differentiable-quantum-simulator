"""Random circuits for property-based and differential tests (no PennyLane needed)."""

import numpy as np

from ketjet.gates import GATES
from ketjet.operations import Operation, Parameter


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


def symbolize(
    circuit: list[Operation], rng: np.random.Generator, share: float = 0.25
) -> tuple[list[Operation], dict[str, float], list[int]]:
    """Replace every parameter by a symbol; a fraction ``share`` of slots reuse an earlier one.

    Returns the symbolic operations, the symbol values, and for each slot (in order) the index of
    its symbol in first-appearance order. A reused symbol keeps its first value, so the symbolic
    circuit bound to the values differs from ``circuit`` at reused slots.
    """
    names: list[str] = []
    values: dict[str, float] = {}
    slot_symbol: list[int] = []
    symbolic = []
    for op in circuit:
        params = []
        for p in op.params:
            if names and rng.random() < share:
                k = int(rng.integers(len(names)))
            else:
                k = len(names)
                names.append(f"p{k}")
                values[f"p{k}"] = float(p)  # type: ignore[arg-type]  # numeric before symbolizing
            slot_symbol.append(k)
            params.append(Parameter(names[k]))
        symbolic.append(Operation(op.gate, op.wires, tuple(params), op.controls, op.control_values))
    return symbolic, values, slot_symbol
