"""An optimization loop converges to the known minimum (Sprint 3 integration test).

Plain gradient descent with adjoint gradients minimizes the energy of a two-qubit transverse-field
Ising Hamiltonian. The exact ground-state energy comes from diagonalizing the dense matrix, which
is independent of the simulator.
"""

import numpy as np

from ketjet import Circuit
from ketjet.gradients import grad
from ketjet.observables import X, Z
from ketjet.operations import Parameter

HAMILTONIAN = Z(0) @ Z(1) + 0.5 * X(0) + 0.5 * X(1)


def _ansatz() -> tuple[Circuit, list[Parameter]]:
    """Hardware-efficient ansatz: RY layer, CNOT, RY layer (spans the real two-qubit states)."""
    p = [Parameter(f"t{k}") for k in range(4)]
    c = Circuit(2).ry(p[0], 0).ry(p[1], 1).cnot(0, 1).ry(p[2], 0).ry(p[3], 1)
    return c, p


def test_gradient_descent_reaches_the_ground_state_energy() -> None:
    exact = float(np.linalg.eigvalsh(HAMILTONIAN.matrix(2))[0])
    circuit, params = _ansatz()
    theta = np.array([0.1, -0.2, 0.3, 0.05])
    energies = []
    for _ in range(300):
        values = dict(zip([p.name for p in params], theta.tolist(), strict=True))
        energies.append(circuit.bind(values).expval(HAMILTONIAN))
        theta = theta - 0.2 * grad(circuit, HAMILTONIAN, values)
    assert energies[-1] - exact < 1e-8
    assert energies[-1] >= exact - 1e-12  # variational: never below the true minimum
    assert all(np.diff(energies) <= 1e-12)  # a small enough step decreases the energy monotonically
