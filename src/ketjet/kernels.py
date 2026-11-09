"""Reference (NumPy) kernels that transform statevectors.

These kernels are the correctness oracle for every faster backend (plan.md §3): they favour clarity
over speed. Conventions follow ADR-0001:

- the state is a flat array of ``2**n`` amplitudes, viewed as a rank-``n`` tensor of shape
  ``(2,) * n`` in which **axis w is wire w**;
- wire 0 is the most significant bit, and the same holds inside a gate matrix: for a gate on wires
  ``(w0, w1, …)``, ``w0`` is the most significant bit of the matrix index;
- kernels are **pure**: they return a new array and never modify their input.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ketjet.statevector import MAX_QUBITS, StateVector, as_complex_dtype

__all__ = ["Wires", "apply_matrix", "n_wires_of"]

Wires = Sequence[int | np.integer[Any]]
"""Target wires: Python or NumPy integers."""


def n_wires_of(state: StateVector) -> int:
    """Return the number of qubits of a flat statevector.

    Parameters
    ----------
    state
        One-dimensional array of ``2**n`` amplitudes.

    Returns
    -------
    int
        ``n``.

    Raises
    ------
    ValueError
        If ``state`` isn't one-dimensional, or its length isn't a power of two of at least 2.
    """
    if state.ndim != 1:
        raise ValueError(f"State must be one-dimensional, got shape {state.shape}.")
    size = int(state.shape[0])
    if size < 2 or size & (size - 1):
        raise ValueError(f"State length must be a power of two >= 2, got {size}.")
    n = size.bit_length() - 1
    if n > MAX_QUBITS:  # pragma: no cover - unallocatable in practice
        raise ValueError(f"State has {n} qubits; at most {MAX_QUBITS} are supported.")
    return n


def _validate_wires(wires: Wires, n_wires: int) -> tuple[int, ...]:
    checked = tuple(wires)
    if not checked:
        raise ValueError("At least one wire is required.")
    for w in checked:
        if isinstance(w, bool) or not isinstance(w, int | np.integer):
            raise TypeError(f"Wires must be ints, got {type(w).__name__}.")
        if not 0 <= w < n_wires:
            raise ValueError(f"Wire {w} is out of range for a {n_wires}-qubit state.")
    if len(set(checked)) != len(checked):
        raise ValueError(f"Wires must be distinct, got {checked}.")
    return tuple(int(w) for w in checked)


def apply_matrix(
    state: StateVector,
    matrix: npt.ArrayLike,
    wires: Wires,
) -> StateVector:
    """Apply a ``2**k x 2**k`` matrix to ``k`` wires of a statevector.

    Parameters
    ----------
    state
        Flat statevector of ``2**n`` amplitudes (complex128 or complex64). It isn't modified.
    matrix
        Square matrix of size ``2**k``, where ``k = len(wires)``. Its first wire is the most
        significant bit of its row and column index. It's cast to the state's dtype, so a complex64
        state stays complex64. Unitarity isn't required: the kernel applies any linear map.
    wires
        The ``k`` distinct target wires, in the order matching the matrix.

    Returns
    -------
    numpy.ndarray
        A new, contiguous statevector with the same shape and dtype as ``state``.

    Raises
    ------
    ValueError
        If the state, matrix shape or wires are invalid.
    TypeError
        If a wire isn't an integer.

    Notes
    -----
    The state is viewed as a tensor of shape ``(2,) * n``. The matrix, reshaped to ``(2,) * 2k``,
    is contracted with the target axes, and the resulting output axes are moved back into the
    target positions. Cost is ``O(2**n * 2**k)`` time and one extra state of memory.

    Examples
    --------
    Flip wire 1 of ``|00⟩`` to get ``|01⟩`` (index 1, since wire 0 is the most significant bit):

    >>> from ketjet import zero_state
    >>> x = [[0, 1], [1, 0]]
    >>> apply_matrix(zero_state(2), x, wires=[1])
    array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])
    """
    dtype = as_complex_dtype(state.dtype)
    n = n_wires_of(state)
    targets = _validate_wires(wires, n)
    k = len(targets)

    op = np.asarray(matrix, dtype=dtype)
    dim = 1 << k
    if op.shape != (dim, dim):
        raise ValueError(f"A gate on {k} wire(s) needs a {dim}x{dim} matrix, got shape {op.shape}.")

    psi = state.reshape((2,) * n)
    op_tensor = op.reshape((2,) * (2 * k))
    # Contract the matrix's input axes (k … 2k-1) with the target axes of the state.
    # tensordot puts the matrix's k output axes first, followed by the untouched state axes.
    out = np.tensordot(op_tensor, psi, axes=(tuple(range(k, 2 * k)), targets))
    out = np.moveaxis(out, tuple(range(k)), targets)
    return np.ascontiguousarray(out, dtype=dtype).reshape(-1)
