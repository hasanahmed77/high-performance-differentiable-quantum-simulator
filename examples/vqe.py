"""A variational quantum eigensolver for the H2 molecule on ketjet.qubit.

The Hamiltonian comes from PennyLane's quantum chemistry module. The ansatz is the
Hartree-Fock state plus one ``DoubleExcitation``, trained with adjoint gradients computed by
ketjet. It reaches the exact (full configuration interaction) ground-state energy.

Run with ``python examples/vqe.py`` after ``pip install "ketjet[pennylane]"``.
"""

from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

STEPS = 40


def vqe() -> float:
    """Optimize the ansatz and return the final energy, in hartree."""
    molecule = qml.qchem.Molecule(
        ["H", "H"],
        np.array([0.0, 0.0, -0.70056, 0.0, 0.0, 0.70056]),  # 0.7414 angstrom, in bohr
        basis_name="sto-3g",
        unit="bohr",
    )
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(molecule, method="dhf")
    hartree_fock = qml.qchem.hf_state(electrons=2, orbitals=n_qubits)
    exact = float(np.linalg.eigvalsh(qml.matrix(hamiltonian, wire_order=range(n_qubits)))[0])

    dev = qml.device("ketjet.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="adjoint")  # type: ignore[untyped-decorator]
    def energy(theta: Any) -> Any:
        qml.BasisState(hartree_fock, wires=range(n_qubits))
        qml.DoubleExcitation(theta[0], wires=range(n_qubits))
        return qml.expval(hamiltonian)

    theta = pnp.array([0.0], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    print(f"Hartree-Fock energy: {energy(theta):.8f} Ha")
    for step in range(STEPS):
        theta = opt.step(energy, theta)
        if step % 10 == 9:
            print(f"step {step + 1:2d}: {energy(theta):.8f} Ha")
    final = float(energy(theta))
    print(f"VQE energy:          {final:.8f} Ha")
    print(f"exact (FCI) energy:  {exact:.8f} Ha  (error {abs(final - exact):.1e})")
    assert abs(final - exact) < 1e-6
    return final


if __name__ == "__main__":
    vqe()
