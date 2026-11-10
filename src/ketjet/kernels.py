"""Reference (NumPy) kernels that transform statevectors.

These kernels are the correctness oracle for every faster backend (plan.md §3): they favour clarity
over speed. Conventions follow ADR-0001:

- the state is a flat array of ``2**n`` amplitudes, viewed as a rank-``n`` tensor of shape
  ``(2,) * n`` in which **axis w is wire w**;
- wire 0 is the most significant bit, and the same holds inside a gate matrix: for a gate on wires
  ``(w0, w1, …)``, ``w0`` is the most significant bit of the matrix index;
- kernels are **pure**: they return a new array and never modify their input.

:func:`apply_matrix` is the dense kernel and handles any matrix. :func:`apply_diagonal` and
:func:`apply_permutation` handle diagonal and phased-permutation gates in one pass over the state,
touching only the amplitudes the gate changes, and :func:`apply_structured` picks between the
three from a gate's structure flags (QS-205).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ketjet.statevector import MAX_QUBITS, StateVector, as_complex_dtype

__all__ = [
    "Wires",
    "apply_diagonal",
    "apply_matrix",
    "apply_permutation",
    "apply_structured",
    "n_wires_of",
]

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


def _basis_index(j: int, targets: tuple[int, ...], n: int) -> tuple[int | slice, ...]:
    """Index selecting basis value ``j`` of ``targets`` (first = MSB) in a :func:`_tensor` view.

    The view's trailing length-1 axis keeps the result a view even when every wire is a target.
    """
    index: list[int | slice] = [slice(None)] * (n + 1)
    k = len(targets)
    for position, wire in enumerate(targets):
        index[wire] = (j >> (k - 1 - position)) & 1
    return tuple(index)


def _tensor(state: StateVector, n: int) -> StateVector:
    return state.reshape((2,) * n + (1,))


def apply_diagonal(
    state: StateVector,
    diagonal: npt.ArrayLike,
    wires: Wires,
) -> StateVector:
    """Apply the diagonal matrix ``diag(diagonal)`` to ``k`` wires of a statevector.

    Equivalent to ``apply_matrix(state, np.diag(diagonal), wires)``, in one pass over the state:
    the state is copied, then only the slices whose diagonal entry isn't 1 are scaled. A
    controlled phase therefore touches only its active block.

    Parameters
    ----------
    state
        Flat statevector of ``2**n`` amplitudes. It isn't modified.
    diagonal
        The ``2**k`` diagonal entries, first wire most significant. Cast to the state's dtype.
    wires
        The ``k`` distinct target wires.

    Returns
    -------
    numpy.ndarray
        A new, contiguous statevector with the same shape and dtype as ``state``.

    Raises
    ------
    ValueError
        If the state, the diagonal's shape or the wires are invalid.
    TypeError
        If a wire isn't an integer.

    Examples
    --------
    CZ on ``|11⟩`` flips its sign:

    >>> import numpy as np
    >>> apply_diagonal(np.array([0, 0, 0, 1], dtype=complex), [1, 1, 1, -1], wires=[0, 1])
    array([ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j])
    """
    dtype = as_complex_dtype(state.dtype)
    n = n_wires_of(state)
    targets = _validate_wires(wires, n)
    d = np.asarray(diagonal, dtype=dtype)
    dim = 1 << len(targets)
    if d.shape != (dim,):
        raise ValueError(
            f"A gate on {len(targets)} wire(s) needs {dim} diagonal entries, got shape {d.shape}."
        )
    out = np.array(state, dtype=dtype, order="C")  # always a copy
    psi = _tensor(out, n)
    for j in np.flatnonzero(d != 1):
        psi[_basis_index(int(j), targets, n)] *= d[j]
    return out


def apply_permutation(
    state: StateVector,
    permutation: npt.ArrayLike,
    phases: npt.ArrayLike,
    wires: Wires,
) -> StateVector:
    r"""Apply a phased permutation, :math:`U|j\rangle = p_j\,|\pi(j)\rangle`, to ``k`` wires.

    Equivalent to ``apply_matrix(state, M, wires)`` where column ``j`` of ``M`` has the single
    entry ``phases[j]`` in row ``permutation[j]``. The amplitudes are moved, not mixed: the state
    is copied, then only the slices the gate changes are rewritten. CNOT touches half the state.

    Parameters
    ----------
    state
        Flat statevector of ``2**n`` amplitudes. It isn't modified.
    permutation
        :math:`\pi`: a permutation of ``0 … 2**k - 1``, first wire most significant.
    phases
        The ``2**k`` non-zero entries :math:`p_j`. Cast to the state's dtype.
    wires
        The ``k`` distinct target wires.

    Returns
    -------
    numpy.ndarray
        A new, contiguous statevector with the same shape and dtype as ``state``.

    Raises
    ------
    ValueError
        If the state or wires are invalid, ``permutation`` isn't a permutation of the right
        length, or ``phases`` has the wrong shape.
    TypeError
        If a wire isn't an integer.

    Examples
    --------
    CNOT maps ``|10⟩`` (index 2) to ``|11⟩`` (index 3):

    >>> import numpy as np
    >>> state = np.array([0, 0, 1, 0], dtype=complex)
    >>> apply_permutation(state, [0, 1, 3, 2], [1, 1, 1, 1], wires=[0, 1])
    array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
    """
    dtype = as_complex_dtype(state.dtype)
    n = n_wires_of(state)
    targets = _validate_wires(wires, n)
    dim = 1 << len(targets)
    perm = np.asarray(permutation)
    if (
        perm.shape != (dim,)
        or perm.dtype.kind not in "iu"
        or not np.array_equal(np.sort(perm), np.arange(dim))
    ):
        raise ValueError(f"permutation must be a permutation of 0 … {dim - 1}, got {perm}.")
    p = np.asarray(phases, dtype=dtype)
    if p.shape != (dim,):
        raise ValueError(f"A gate on {len(targets)} wire(s) needs {dim} phases, got {p.shape}.")
    source = _tensor(state, n)
    out = np.array(state, dtype=dtype, order="C")  # always a copy
    psi = _tensor(out, n)
    for j in range(dim):
        row = int(perm[j])
        if row == j and p[j] == 1:
            continue  # this slice is unchanged
        destination = psi[_basis_index(row, targets, n)]
        np.multiply(source[_basis_index(j, targets, n)], p[j], out=destination, casting="unsafe")
    return out


def apply_structured(
    state: StateVector,
    matrix: npt.ArrayLike,
    wires: Wires,
    *,
    diagonal: bool = False,
    permutation: bool = False,
) -> StateVector:
    """Apply ``matrix`` with the fastest kernel its declared structure allows.

    Parameters
    ----------
    state, matrix, wires
        As for :func:`apply_matrix`.
    diagonal
        ``matrix`` is diagonal: use :func:`apply_diagonal`. Takes precedence over
        ``permutation`` (a diagonal unitary is also a phased permutation).
    permutation
        ``matrix`` has exactly one non-zero entry in every row and column: use
        :func:`apply_permutation`.

    Returns
    -------
    numpy.ndarray
        The same result as ``apply_matrix(state, matrix, wires)``, up to rounding.

    Notes
    -----
    The flags are trusted, not checked: a wrong flag gives a wrong result. Gate flags are verified
    against their matrices by the test suite (ADR-0002 D3).
    """
    if not (diagonal or permutation):
        return apply_matrix(state, matrix, wires)
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {m.shape}.")
    if diagonal:
        return apply_diagonal(state, np.diagonal(m), wires)
    rows = np.argmax(np.abs(m), axis=0)  # the non-zero entry of each column
    return apply_permutation(state, rows, m[rows, np.arange(m.shape[1])], wires)
