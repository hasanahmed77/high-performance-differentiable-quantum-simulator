"""A Bell state on ketjet.qubit: exact probabilities, a correlation, and seeded samples.

Run with ``python examples/bell.py`` after ``pip install "ketjet[pennylane]"``.
"""

from typing import Any

import numpy as np
import pennylane as qml

SHOTS = 1000


def bell() -> None:
    """Prepare (|00> + |11>) / sqrt(2) and measure it exactly and with shots."""
    dev = qml.device("ketjet.qubit", wires=2, seed=7)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]  # PennyLane ships no type hints
    def exact() -> Any:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1]), qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    probs, correlation = exact()
    print("probabilities |00>, |01>, |10>, |11>:", probs.round(3))
    print(f"<Z0 Z1> = {correlation:.3f}  (the qubits always agree)")
    assert np.allclose(probs, [0.5, 0, 0, 0.5])
    assert np.isclose(correlation, 1.0)

    @qml.set_shots(SHOTS)  # type: ignore[untyped-decorator]
    @qml.qnode(dev)  # type: ignore[untyped-decorator]
    def sampled() -> Any:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(wires=[0, 1])

    counts = {str(k): int(v) for k, v in sampled().items()}
    print(f"{SHOTS} shots:", dict(sorted(counts.items())))
    assert set(counts) <= {"00", "11"}  # never 01 or 10
    assert sum(counts.values()) == SHOTS
    assert abs(counts.get("00", 0) - SHOTS / 2) <= 4 * np.sqrt(SHOTS) / 2  # within 4 sigma


if __name__ == "__main__":
    bell()
