"""The quantum Fourier transform on ketjet.qubit, checked against its formula.

PennyLane's ``qml.QFT`` template isn't a ketjet gate: the device decomposes it into Hadamards,
controlled phase shifts and SWAPs. The result must equal
QFT|j> = sum_k exp(2 pi i j k / 2^n) |k> / sqrt(2^n).

Run with ``python examples/qft.py`` after ``pip install "ketjet[pennylane]"``.
"""

from typing import Any

import numpy as np
import pennylane as qml

N_WIRES = 4
BASIS_STATE = 0b0110


def qft() -> None:
    """Apply the QFT to |0110> and compare the state with the analytic formula."""
    dev = qml.device("ketjet.qubit", wires=N_WIRES)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]  # PennyLane ships no type hints
    def circuit() -> Any:
        qml.BasisState(
            np.array([int(b) for b in f"{BASIS_STATE:0{N_WIRES}b}"]), wires=range(N_WIRES)
        )
        qml.QFT(wires=range(N_WIRES))
        return qml.state()

    state = circuit()
    size = 2**N_WIRES
    expected = np.exp(2j * np.pi * BASIS_STATE * np.arange(size) / size) / np.sqrt(size)
    error = np.max(np.abs(state - expected))
    label = f"{BASIS_STATE:0{N_WIRES}b}"
    print(f"QFT|{label}> on {N_WIRES} qubits: max deviation from the formula = {error:.1e}")
    print(
        "phases of the first four amplitudes (units of pi):", (np.angle(state[:4]) / np.pi).round(3)
    )
    assert error < 1e-12
    assert np.allclose(np.abs(state) ** 2, 1 / size)  # a uniform superposition


if __name__ == "__main__":
    qft()
