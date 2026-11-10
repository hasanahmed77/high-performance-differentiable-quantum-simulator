"""Differential tests: observable matrices equal PennyLane's (QS-305, ADR-0003)."""

from typing import Any

import numpy as np
import pytest

from ketjet.observables import Hermitian, I, Observable, X, Y, Z
from tests.tolerances import assert_close

qml = pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

rng = np.random.default_rng(11)
_a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
H4 = (_a + _a.conj().T) / 2
_b = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
H2 = (_b + _b.conj().T) / 2

CASES: list[tuple[str, Observable, Any]] = [
    ("Z0", Z(0), lambda: qml.PauliZ(0)),
    ("X2", X(2), lambda: qml.PauliX(2)),
    ("Y1", Y(1), lambda: qml.PauliY(1)),
    ("Z0 Z1", Z(0) @ Z(1), lambda: qml.PauliZ(0) @ qml.PauliZ(1)),
    ("X0 Y1 Z2", X(0) @ Y(1) @ Z(2), lambda: qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)),
    ("identity", I(), lambda: qml.Identity(0)),
    ("Hermitian(1,0)", Hermitian(H4, [1, 0]), lambda: qml.Hermitian(H4, wires=[1, 0])),
    (
        "Hermitian ⊗ X",
        Hermitian(H2, [2]) @ X(0),
        lambda: qml.Hermitian(H2, wires=2) @ qml.PauliX(0),
    ),
    (
        "Hamiltonian",
        0.5 * Z(0) @ Z(1) - 0.2 * X(0) + 1.3 * Y(1) @ X(2),
        lambda: (
            0.5 * qml.PauliZ(0) @ qml.PauliZ(1)
            - 0.2 * qml.PauliX(0)
            + 1.3 * qml.PauliY(1) @ qml.PauliX(2)
        ),
    ),
    (
        "mixed Hamiltonian",
        2.0 * Hermitian(H4, [0, 2]) - Z(1) + 0.25 * I(),
        lambda: 2.0 * qml.Hermitian(H4, wires=[0, 2]) - qml.PauliZ(1) + 0.25 * qml.Identity(0),
    ),
]


@pytest.mark.parametrize(("name", "obs", "pl_obs"), CASES, ids=[c[0] for c in CASES])
def test_observable_matrix_matches_pennylane(name: str, obs: Observable, pl_obs: Any) -> None:
    expected = qml.matrix(pl_obs(), wire_order=[0, 1, 2])
    assert_close(obs.matrix(3), expected, np.complex128)
