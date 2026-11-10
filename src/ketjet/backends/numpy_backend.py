"""The NumPy reference backend: the correctness oracle for every faster backend (plan.md §3)."""

from collections.abc import Iterable, Sequence

import numpy as np

from ketjet import measurements
from ketjet.kernels import apply_matrix, n_wires_of
from ketjet.measurements import Probabilities, Samples, Seed
from ketjet.observables import Observable
from ketjet.operations import Operation
from ketjet.statevector import DTypeLike, StateVector, zero_state

__all__ = ["NumPyBackend"]


class NumPyBackend:
    """Pure NumPy statevector backend.

    Every method is pure: it never modifies the state it receives, which makes it the simplest
    possible implementation of the :class:`~ketjet.backends.Backend` protocol. Callers should still
    follow the protocol's ownership rule and only use returned states, so that switching to an
    in-place backend never changes their results.

    Examples
    --------
    >>> from ketjet.operations import CNOT, H
    >>> backend = NumPyBackend()
    >>> bell = backend.apply_operations(backend.zero_state(2), [H(0), CNOT(0, 1)])
    >>> backend.to_numpy(bell).round(3)
    array([0.707+0.j, 0.   +0.j, 0.   +0.j, 0.707+0.j])
    """

    name = "numpy"

    def zero_state(self, n_wires: int, dtype: DTypeLike = np.complex128) -> StateVector:
        """Allocate the ``|0…0⟩`` state of ``n_wires`` qubits."""
        return zero_state(n_wires, dtype)

    def n_wires(self, state: StateVector) -> int:
        """Return the number of qubits of ``state``."""
        return n_wires_of(state)

    def apply_operation(self, state: StateVector, op: Operation) -> StateVector:
        """Apply one operation, controls included, and return the new state.

        Raises
        ------
        ValueError
            If the operation uses a wire outside the state, or has unbound symbolic parameters.
        """
        n = n_wires_of(state)
        out_of_range = [w for w in op.all_wires if w >= n]
        if out_of_range:
            raise ValueError(
                f"{op.gate} uses wire(s) {out_of_range}, but the state has {n} qubit(s)."
            )
        return apply_matrix(state, op.matrix(state.dtype), op.all_wires)

    def apply_operations(self, state: StateVector, ops: Iterable[Operation]) -> StateVector:
        """Apply operations in order and return the final state."""
        for op in ops:
            state = self.apply_operation(state, op)
        return state

    def to_numpy(self, state: StateVector) -> StateVector:
        """Return a copy of the amplitudes (the caller may modify it freely)."""
        return state.copy()

    def probs(self, state: StateVector, wires: Sequence[int] | None = None) -> Probabilities:
        """Return measurement probabilities of ``wires`` (see :func:`ketjet.measurements.probs`)."""
        return measurements.probs(state, wires)

    def expval(self, state: StateVector, observable: Observable) -> float:
        """Return the expectation value of ``observable``."""
        return measurements.expval(state, observable)

    def var(self, state: StateVector, observable: Observable) -> float:
        """Return the variance of ``observable``."""
        return measurements.var(state, observable)

    def sample(
        self, state: StateVector, shots: int, wires: Sequence[int] | None = None, seed: Seed = None
    ) -> Samples:
        """Return measured bits (see :func:`ketjet.measurements.sample`)."""
        return measurements.sample(state, shots, wires, seed)
