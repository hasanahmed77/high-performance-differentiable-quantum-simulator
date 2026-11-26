"""ketjet's nightly regression benchmarks for asv (QS-1105, ADR-0008 D7).

asv compares two commits of ketjet in one job (``asv continuous``), so the hosted runner's own
speed cancels out. These benchmarks use only ketjet's public API, because asv runs them against
older commits too, and they build their circuits in ``setup``, outside the timer. They're small
enough for a hosted runner: the published cross-library numbers come from ``benchmarks/`` on the
workstation, not from here.
"""

from typing import Any, ClassVar

import numpy as np

from ketjet import gradients
from ketjet import operations as ops
from ketjet.circuit import Circuit
from ketjet.observables import Observable, X, Z
from ketjet.operations import Parameter

SIZES = [12, 16]
DEPTH = 8


def _random(n: int) -> Circuit:
    """Return a seeded circuit of ``DEPTH`` layers: a rotation per wire, then CNOTs on pairs."""
    rng = np.random.default_rng(n)
    gates = (ops.RX, ops.RY, ops.RZ)
    circuit = Circuit(n)
    for _ in range(DEPTH):
        for w in range(n):
            circuit.append(gates[int(rng.integers(3))](float(rng.uniform(-np.pi, np.pi)), w))
        wires = rng.permutation(n)
        circuit.extend(
            ops.CNOT(int(a), int(b)) for a, b in zip(wires[::2], wires[1::2], strict=False)
        )
    return circuit


def _ansatz(n: int) -> tuple[Circuit, Observable, dict[str, float]]:
    """Return a 4-layer hardware-efficient ansatz, a transverse-field Ising energy and values."""
    circuit = Circuit(n)
    for layer in range(4):
        for w in range(n):
            circuit.append(ops.RY(Parameter(f"y{layer}_{w}"), w))
            circuit.append(ops.RZ(Parameter(f"z{layer}_{w}"), w))
        circuit.extend(ops.CNOT(w, (w + 1) % n) for w in range(n))
    hamiltonian: Observable = Z(0) @ Z(1)
    for w in range(1, n - 1):
        hamiltonian = hamiltonian + Z(w) @ Z(w + 1)
    for w in range(n):
        hamiltonian = hamiltonian + 0.5 * X(w)
    rng = np.random.default_rng(n)
    values = {
        p.name: float(v)
        for p, v in zip(circuit.parameters, rng.uniform(-np.pi, np.pi, 8 * n), strict=True)
    }
    return circuit, hamiltonian, values


class NumPyState:
    """The NumPy backend's final state of a random circuit."""

    params: ClassVar = SIZES
    param_names: ClassVar = ["qubits"]

    def setup(self, n: int) -> None:
        """Build the circuit."""
        self.circuit = _random(n)

    def time_state(self, n: int) -> None:
        """Time ``Circuit.state()``."""
        self.circuit.state()


class NumPyAdjoint:
    """The NumPy backend's adjoint gradient of the Ising energy."""

    params: ClassVar = SIZES
    param_names: ClassVar = ["qubits"]

    def setup(self, n: int) -> None:
        """Build the ansatz."""
        self.circuit, self.observable, self.values = _ansatz(n)

    def time_gradient(self, n: int) -> None:
        """Time ``gradients.grad`` with the adjoint method."""
        gradients.grad(self.circuit, self.observable, self.values)


class JAXState:
    """The JAX backend's compiled final state of a random circuit, after compiling."""

    params: ClassVar = SIZES
    param_names: ClassVar = ["qubits"]

    def setup(self, n: int) -> None:
        """Compile the circuit and make the first call."""
        import jax  # noqa: PLC0415 - JAX is optional

        from ketjet.backends import jax_backend  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
        circuit = _random(n)
        run = jax_backend.compile_circuit(circuit, dtype="complex128")
        x = jax.numpy.asarray(jax_backend.parameter_values(circuit))
        self.call: Any = lambda: run(x).block_until_ready()
        self.call()

    def time_state(self, n: int) -> None:
        """Time one compiled call."""
        self.call()


class JAXBackprop:
    """The JAX backend's backprop gradient of the Ising energy, after compiling."""

    params: ClassVar = SIZES
    param_names: ClassVar = ["qubits"]

    def setup(self, n: int) -> None:
        """Compile the gradient and make the first call."""
        import jax  # noqa: PLC0415

        from ketjet.backends import jax_backend  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
        circuit, observable, values = _ansatz(n)
        f = jax_backend.expval_fn(circuit, [observable], dtype="complex128")
        grad = jax.jit(jax.grad(lambda v: f(v)[0]))
        weights = jax.numpy.asarray([values[p.name] for p in circuit.parameters])
        self.call: Any = lambda: grad(weights).block_until_ready()
        self.call()

    def time_gradient(self, n: int) -> None:
        """Time one compiled gradient."""
        self.call()
