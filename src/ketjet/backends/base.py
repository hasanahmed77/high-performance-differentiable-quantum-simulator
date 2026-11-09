"""The backend protocol every simulator backend implements (ADR-0001 D4).

A backend owns the representation of a state (a NumPy array, a JAX array, a native buffer, a GPU
buffer) and the kernels that transform it. Callers interact only through this protocol, so the same
test suite runs against every backend.

**Ownership rule (ADR-0001 D4):** methods that transform a state *consume* their input. Callers
must use the returned state and must not read or reuse the state they passed in. Pure backends
(NumPy, JAX) return a new state; in-place backends (C++, CUDA) may update the buffer and return it,
avoiding a second ``2**n`` allocation.
"""

from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from ketjet.operations import Operation
from ketjet.statevector import DTypeLike, StateVector

__all__ = ["Backend", "StateT"]

StateT = TypeVar("StateT")
"""The backend's native state type."""


@runtime_checkable
class Backend(Protocol[StateT]):
    """Interface implemented by every simulator backend.

    Attributes
    ----------
    name
        Short identifier, e.g. ``"numpy"``.
    """

    name: str

    def zero_state(self, n_wires: int, dtype: DTypeLike = ...) -> StateT:
        """Allocate the ``|0…0⟩`` state of ``n_wires`` qubits."""
        ...

    def n_wires(self, state: StateT) -> int:
        """Return the number of qubits of ``state``."""
        ...

    def apply_operation(self, state: StateT, op: Operation) -> StateT:
        """Apply one operation. Consumes ``state``; use the returned state."""
        ...

    def apply_operations(self, state: StateT, ops: Iterable[Operation]) -> StateT:
        """Apply operations in order. Consumes ``state``; use the returned state."""
        ...

    def to_numpy(self, state: StateT) -> StateVector:
        """Return the amplitudes as a flat NumPy array (wire 0 = most significant bit)."""
        ...
