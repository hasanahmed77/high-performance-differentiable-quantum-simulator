"""Canonical circuits pinned by the regression golden files."""

import numpy as np

from ketjet import Circuit
from ketjet import operations as ops
from ketjet.observables import Observable, X, Y, Z
from ketjet.operations import Parameter


def qft(n: int, basis_state: int) -> Circuit:
    """Quantum Fourier transform of the computational basis state ``|basis_state>``."""
    c = Circuit(n)
    for w in range(n):
        if (basis_state >> (n - 1 - w)) & 1:
            c.x(w)
    for target in range(n):
        c.h(target)
        for k, control in enumerate(range(target + 1, n), start=2):
            c.append(ops.ctrl(ops.PhaseShift(2 * np.pi / 2**k, target), [control]))
    for w in range(n // 2):
        c.swap(w, n - 1 - w)
    return c


def ghz(n: int) -> Circuit:
    """GHZ state (|0...0> + |1...1>) / sqrt(2)."""
    c = Circuit(n).h(0)
    for w in range(n - 1):
        c.cnot(w, w + 1)
    return c


def observables(n: int) -> dict[str, Observable]:
    """Named observables evaluated on every golden circuit."""
    zz: Observable = Z(0)
    for w in range(1, n):
        zz = zz @ Z(w)
    return {
        "Z0": Z(0),
        "X0": X(0),
        "Z_all": zz,
        "hamiltonian": 0.5 * Z(0) @ Z(1) - 0.3 * X(0) @ Y(n - 1) + 0.2 * X(n - 1),
    }


CIRCUITS = {"qft4_0110": qft(4, 0b0110), "qft5_10011": qft(5, 0b10011), "ghz5": ghz(5)}


def gradient_circuit() -> tuple[Circuit, dict[str, float]]:
    """Three shared symbols over plain, controlled and multi-parameter gates, with fixed values.

    Every symbol appears in several slots (the chain rule), including controlled rotations (the
    four-term shift rule) and gates after an entangler.
    """
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    circuit = (
        Circuit(3)
        .h(0)
        .ry(a, 1)
        .crx(b, 0, 1)
        .rot(a, c, 0.3, 2)
        .cnot(1, 2)
        .crz(c, 2, 0)
        .phase(b, 1)
        .u3(0.2, a, c, 0)
        .iswap(0, 2)
        .cry(b, 1, 2)
    )
    return circuit, {"a": 0.37, "b": -1.21, "c": 2.05}


GRADIENT_CIRCUITS = {"gradient3": gradient_circuit()}
