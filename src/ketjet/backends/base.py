"""The backend protocol every simulator backend implements (ADR-0001 D4).

A backend owns the representation of a state (a NumPy array, a JAX array, a native buffer, a GPU
buffer) and the kernels that transform it. Callers interact only through this protocol, so the same
test suite runs against every backend.

**Ownership rule (ADR-0001 D4):** methods that transform a state *consume* their input. Callers
must use the returned state and must not read or reuse the state they passed in. Pure backends
(NumPy, JAX) return a new state; in-place backends (C++, CUDA) may update the buffer and return it,
avoiding a second ``2**n`` allocation.

**Measurements are read-only (ADR-0003 D6):** ``probs``, ``expval``, ``var`` and ``sample`` don't
consume the state. Backends compute them where the state lives (e.g. on a GPU) and return only
the results.

**Adjoint differentiation is one coarse method (ADR-0004 D4),** so native and GPU backends can run
the whole backward pass on their own memory, without a host call per gate.
"""

from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from ketjet.measurements import Probabilities, Samples, Seed
from ketjet.observables import Observable
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

    def probs(self, state: StateT, wires: Sequence[int] | None = None) -> Probabilities:
        """Return measurement probabilities of ``wires`` (``wires[0]`` = most significant bit)."""
        ...

    def expval(self, state: StateT, observable: Observable) -> float:
        """Return the expectation value of ``observable``."""
        ...

    def var(self, state: StateT, observable: Observable) -> float:
        """Return the variance of ``observable``."""
        ...

    def sample(
        self, state: StateT, shots: int, wires: Sequence[int] | None = None, seed: Seed = None
    ) -> Samples:
        """Return ``(shots, len(wires))`` measured bits, reproducible for a given ``seed``."""
        ...

    def adjoint_jacobian(
        self,
        n_wires: int,
        operations: Sequence[Operation],
        observables: Sequence[Observable],
        trainable: Sequence[tuple[int, int]],
        *,
        dtype: DTypeLike = ...,
    ) -> npt.NDArray[np.float64]:
        """Return ``∂⟨O_m⟩/∂x_s`` for the ``trainable`` ``(operation, parameter)`` slots.

        Shape ``(len(observables), len(trainable))``. The circuit starts from ``|0…0⟩``; memory
        must not grow with circuit depth (see :mod:`ketjet.adjoint`).
        """
        ...
