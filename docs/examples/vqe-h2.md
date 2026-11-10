# Example: the ground state of H₂

This example finds the ground-state energy of the hydrogen molecule with a variational quantum eigensolver
(VQE): a parametrized circuit prepares a trial state, and gradient descent with exact adjoint gradients
minimizes its energy. It reaches the exact answer to full double precision.

Every code block on this page is executed by the test suite (`tests/integration/test_vqe_h2.py`), so the
printed values are the real ones.

## The Hamiltonian

The electronic Hamiltonian of H₂ at its equilibrium bond length (0.7414 Å) in the minimal STO-3G basis,
mapped to four qubits with the Jordan–Wigner transformation, is a sum of 15 Pauli words. The coefficients (in
hartree) come from PennyLane's `qml.qchem.molecular_hamiltonian` with `method="dhf"`. A test regenerates them
and checks they haven't drifted.

```python
import numpy as np

import ketjet as kj
from ketjet import gradients
from ketjet.observables import Pauli, Sum
from ketjet.operations import Parameter

TERMS = {
    "IIII": -0.098916212042699,
    "ZIII": 0.171191490893100,
    "IZII": 0.171191490893100,
    "ZZII": 0.168620246802723,
    "IIZI": -0.222767153082905,
    "ZIZI": 0.120542480410302,
    "IZZI": 0.165865253002722,
    "YXXY": 0.045322772592420,
    "YYXX": -0.045322772592420,
    "XXYY": -0.045322772592420,
    "XYYX": 0.045322772592420,
    "IIIZ": -0.222767153082905,
    "ZIIZ": 0.165865253002722,
    "IZIZ": 0.120542480410302,
    "IIZZ": 0.174346565285259,
}
H = Sum([(c, Pauli([(w, p) for w, p in enumerate(word) if p != "I"])) for word, c in TERMS.items()])

exact = np.linalg.eigvalsh(H.matrix(4))[0]  # full configuration interaction (FCI)
print(f"{exact:.10f}")  # -1.1372697289
```

## The ansatz

Qubits 0 and 1 are the bonding orbital (spin up and down), 2 and 3 the antibonding one. The Hartree–Fock state
puts both electrons in the bonding orbital: $|1100\rangle$. Particle number, spin and symmetry leave one other
configuration the ground state can mix with, $|0011\rangle$, so one parameter is enough:

$$
|\psi(\theta)\rangle = \cos\tfrac{\theta}{2}\,|0011\rangle + \sin\tfrac{\theta}{2}\,|1100\rangle .
$$

An RY rotation and three CNOTs make $\cos\frac{\theta}{2}|0000\rangle + \sin\frac{\theta}{2}|1111\rangle$, and two
X gates flip the antibonding qubits. At $\theta = \pi$ it's the Hartree–Fock state.

```python
theta = Parameter("theta")
ansatz = kj.Circuit(4).ry(theta, 0).cnot(0, 1).cnot(1, 2).cnot(1, 3).x(2).x(3)

hartree_fock = ansatz.bind({"theta": np.pi}).expval(H)
print(f"{hartree_fock:.10f}")  # -1.1166821547
```

## Optimizing with adjoint gradients

`gradients.grad` uses the adjoint method by default: one forward and one backward pass, whatever the number of
parameters.

```python
values = {"theta": np.pi}  # start from Hartree–Fock
energies = []
for step in range(40):
    energies.append(ansatz.bind(values).expval(H))
    values["theta"] -= 0.5 * gradients.grad(ansatz, H, values)[0]
energy = ansatz.bind(values).expval(H)

print(f"{energy:.10f}")  # -1.1372697289
print(abs(energy - exact) < 1e-12)  # True: exact to double precision
```

The correlation energy, the part Hartree–Fock misses, is about 20.6 millihartree, and VQE recovers all of it:

```python
print(f"{(hartree_fock - exact) * 1000:.2f} mHa")  # 20.59 mHa
```

## The three methods agree

At the optimum the gradient vanishes. Away from it, adjoint, parameter-shift (the method that also works on
hardware) and finite differences give the same derivative. Finite differences are accurate to about $10^{-9}$
here, so six decimals is as far as all three agree on every platform:

```python
point = {"theta": 2.0}
for method in ("adjoint", "parameter-shift", "finite-diff"):
    g = gradients.grad(ansatz, H, point, method=method)[0]
    print(f"{method:>15}: {g:.6f}")
#         adjoint: -0.791895
# parameter-shift: -0.791895
#     finite-diff: -0.791895
```
