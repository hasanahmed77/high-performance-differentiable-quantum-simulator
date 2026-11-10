"""Canonical circuits pinned by the regression golden files."""

import numpy as np

from ketjet import Circuit
from ketjet import operations as ops
from ketjet.observables import Observable, X, Y, Z


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
