"""The benchmark circuits (QS-1102): one spec per benchmark, built as a ketjet circuit.

Every circuit is symbolic, with seeded values, so each library gets the same gates and the same
angles (ADR-0008 D5). Circuits that would otherwise start from a constant state begin with a
layer of ``RY`` on every wire, with its angles as parameters, so a compiler can't evaluate the
circuit ahead of time.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

from ketjet import operations as ops
from ketjet.circuit import Circuit
from ketjet.observables import Observable, X, Z
from ketjet.operations import Parameter

GATE_LAYERS: Final = 10
"""Layers of the benchmarked gate in each single-gate benchmark."""

RANDOM_DEPTH: Final = 8
"""Layers of a random circuit: one random rotation per wire, then CNOTs on random pairs."""

VQE_LAYERS: Final = 4
"""Layers of the hardware-efficient VQE ansatz."""

QML_LAYERS: Final = 2
"""Trainable layers of the QML classifier."""

QML_BATCH: Final = 32
"""Inputs in each batch of the QML benchmark."""


@dataclass(frozen=True)
class Spec:
    """One benchmark circuit at one size.

    Attributes
    ----------
    circuit
        The symbolic circuit.
    values
        One value per symbol, in :attr:`Circuit.parameters <ketjet.circuit.Circuit.parameters>`
        order. For the QML benchmark these are the weights only.
    observable
        The measured observable of gradient benchmarks; ``None`` for state benchmarks.
    inputs
        The QML benchmark's batch of inputs, shape ``(QML_BATCH, n_qubits)``. The circuit's
        first ``n_qubits`` symbols are the inputs, the rest the weights.
    """

    circuit: Circuit
    values: npt.NDArray[np.float64]
    observable: Observable | None = None
    inputs: npt.NDArray[np.float64] | None = field(default=None)

    @property
    def gates(self) -> int:
        """Number of gates in the circuit."""
        return len(self.circuit)

    def bound(self) -> Circuit:
        """Return the circuit with :attr:`values` bound (state benchmarks only)."""
        return self.circuit.bind(dict(zip(self.circuit.parameters, self.values, strict=True)))


def _rng(name: str, n: int) -> np.random.Generator:
    """Return a generator seeded by the benchmark's name and size, so every run agrees."""
    return np.random.default_rng([sum(map(ord, name)), n])


def _angles(rng: np.random.Generator, count: int) -> npt.NDArray[np.float64]:
    return rng.uniform(-np.pi, np.pi, count)


def _prepare(n: int) -> Circuit:
    return Circuit(n, (ops.RY(Parameter(f"p{w}"), w) for w in range(n)))


def gate_layers(gate: str, n: int) -> Spec:
    """``GATE_LAYERS`` layers of one gate on every wire, after an ``RY`` layer.

    ``gate`` is ``"h"`` (a dense single-qubit gate), ``"rz"`` (diagonal, with fixed angles) or
    ``"cnot"`` (a permutation, on neighbouring pairs, alternating between even and odd pairs).
    """
    rng = _rng(f"gate-{gate}", n)
    circuit = _prepare(n)
    for layer in range(GATE_LAYERS):
        if gate == "h":
            circuit.extend(ops.H(w) for w in range(n))
        elif gate == "rz":
            circuit.extend(ops.RZ(float(a), w) for w, a in enumerate(_angles(rng, n)))
        elif gate == "cnot":
            circuit.extend(ops.CNOT(w, w + 1) for w in range(layer % 2, n - 1, 2))
        else:
            raise ValueError(f"Unknown gate {gate!r}; use 'h', 'rz' or 'cnot'.")
    return Spec(circuit, _angles(rng, n))


def qft(n: int) -> Spec:
    """Return the quantum Fourier transform (H, controlled phases, SWAPs) after an RY layer."""
    rng = _rng("qft", n)
    circuit = _prepare(n)
    for target in range(n):
        circuit.append(ops.H(target))
        for k, control in enumerate(range(target + 1, n), start=2):
            circuit.append(ops.ctrl(ops.PhaseShift(2 * np.pi / 2**k, target), [control]))
    circuit.extend(ops.SWAP(w, n - 1 - w) for w in range(n // 2))
    return Spec(circuit, _angles(rng, n))


def random_circuit(n: int) -> Spec:
    """``RANDOM_DEPTH`` layers of a random rotation per wire and CNOTs on random pairs."""
    rng = _rng("random", n)
    rotations: tuple[Callable[[Parameter, int], ops.Operation], ...] = (ops.RX, ops.RY, ops.RZ)
    circuit = Circuit(n)
    for layer in range(RANDOM_DEPTH):
        for w in range(n):
            gate = rotations[int(rng.integers(3))]
            circuit.append(gate(Parameter(f"t{layer}_{w}"), w))
        wires = rng.permutation(n)
        circuit.extend(
            ops.CNOT(int(a), int(b)) for a, b in zip(wires[::2], wires[1::2], strict=False)
        )
    return Spec(circuit, _angles(rng, len(circuit.parameters)))


def ising(n: int) -> Observable:
    """Return the transverse-field Ising Hamiltonian: the sum of ``Z_i Z_{i+1}`` and ``0.5 X_i``."""
    terms: list[Observable] = [Z(w) @ Z(w + 1) for w in range(n - 1)]
    terms += [0.5 * X(w) for w in range(n)]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def vqe_ansatz(n: int) -> Spec:
    """Return a hardware-efficient ansatz (RY, RZ per wire, a CNOT ring) measuring :func:`ising`."""
    rng = _rng("vqe", n)
    circuit = Circuit(n)
    for layer in range(VQE_LAYERS):
        for w in range(n):
            circuit.append(ops.RY(Parameter(f"y{layer}_{w}"), w))
            circuit.append(ops.RZ(Parameter(f"z{layer}_{w}"), w))
        circuit.extend(ops.CNOT(w, (w + 1) % n) for w in range(n if n > 2 else 1))
    return Spec(circuit, _angles(rng, len(circuit.parameters)), ising(n))


def qml_classifier(n: int) -> Spec:
    """Return a classifier: RX angle encoding of ``n`` features, then trainable RY and CNOT layers.

    It measures ``Z(0)`` on a batch of ``QML_BATCH`` inputs; the benchmark is the gradient of the
    batch's mean with respect to the weights.
    """
    rng = _rng("qml", n)
    circuit = Circuit(n, (ops.RX(Parameter(f"x{w}"), w) for w in range(n)))
    for layer in range(QML_LAYERS):
        circuit.extend(ops.RY(Parameter(f"w{layer}_{w}"), w) for w in range(n))
        circuit.extend(ops.CNOT(w, w + 1) for w in range(n - 1))
    weights = _angles(rng, QML_LAYERS * n)
    inputs = rng.uniform(0, np.pi, (QML_BATCH, n))
    return Spec(circuit, weights, Z(0), inputs)


BENCHMARKS: Final[dict[str, Callable[[int], Spec]]] = {
    "gate-h": lambda n: gate_layers("h", n),
    "gate-rz": lambda n: gate_layers("rz", n),
    "gate-cnot": lambda n: gate_layers("cnot", n),
    "qft": qft,
    "random": random_circuit,
    "vqe-grad": vqe_ansatz,
    "qml-batch": qml_classifier,
}
"""Every benchmark by name, as a function of the number of qubits."""

GRADIENT_BENCHMARKS: Final = frozenset({"vqe-grad", "qml-batch"})
"""Benchmarks that time a gradient; the others time the final state."""
