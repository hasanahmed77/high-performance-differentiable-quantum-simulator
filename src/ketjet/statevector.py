"""Statevector allocation, precision handling and memory accounting.

Conventions follow ADR-0001:

- A state of ``n`` qubits is a contiguous, C-ordered 1-D array of ``2**n`` complex amplitudes.
- Wire 0 is the most significant bit of the basis index.
- Supported precisions are ``complex128`` (default) and ``complex64``; a state's dtype is fixed at
  allocation and is never silently promoted.
"""

from typing import Any, Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "MAX_QUBITS",
    "SUPPORTED_DTYPES",
    "ComplexDType",
    "DTypeLike",
    "StateVector",
    "as_complex_dtype",
    "state_nbytes",
    "validate_n_wires",
    "zero_state",
]

MAX_QUBITS: Final = 63
"""Largest supported qubit count: ``2**63`` amplitudes is the limit of a 64-bit index."""

ComplexDType = np.dtype[np.complexfloating[Any, Any]]
"""A NumPy complex dtype (``complex64`` or ``complex128``)."""

StateVector = npt.NDArray[np.complexfloating[Any, Any]]
"""A statevector: 1-D array of ``2**n`` complex amplitudes."""

DTypeLike = type[np.complex64] | type[np.complex128] | np.dtype[Any] | str
"""Accepted precision spellings: ``np.complex128``, ``np.dtype("complex64")`` or ``"complex64"``."""

SUPPORTED_DTYPES: Final[tuple[ComplexDType, ...]] = (
    np.dtype(np.complex128),
    np.dtype(np.complex64),
)


def as_complex_dtype(dtype: DTypeLike) -> ComplexDType:
    """Normalize a precision specification to a supported complex dtype.

    Parameters
    ----------
    dtype
        ``np.complex128``, ``np.complex64``, a matching ``np.dtype`` or its name.

    Returns
    -------
    numpy.dtype
        ``dtype('complex128')`` or ``dtype('complex64')``.

    Raises
    ------
    ValueError
        If ``dtype`` isn't one of the supported precisions.
    """
    try:
        normalized = np.dtype(dtype)
    except TypeError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected complex128 or complex64.") from exc
    for supported in SUPPORTED_DTYPES:
        if normalized == supported:
            return supported
    raise ValueError(f"Unsupported dtype {normalized}; expected complex128 or complex64.")


def validate_n_wires(n_wires: int) -> int:
    """Check that ``n_wires`` is a valid qubit count.

    Parameters
    ----------
    n_wires
        Number of qubits.

    Returns
    -------
    int
        ``n_wires``, unchanged.

    Raises
    ------
    TypeError
        If ``n_wires`` isn't an integer (``bool`` is rejected too).
    ValueError
        If ``n_wires`` is outside ``1 … MAX_QUBITS``.
    """
    if isinstance(n_wires, bool) or not isinstance(n_wires, int | np.integer):
        raise TypeError(f"n_wires must be an int, got {type(n_wires).__name__}.")
    if not 1 <= n_wires <= MAX_QUBITS:
        raise ValueError(f"n_wires must be between 1 and {MAX_QUBITS}, got {n_wires}.")
    return int(n_wires)


def state_nbytes(n_wires: int, dtype: DTypeLike = np.complex128) -> int:
    """Return the memory needed for an ``n_wires``-qubit statevector, in bytes.

    Parameters
    ----------
    n_wires
        Number of qubits.
    dtype
        Precision of the state.

    Returns
    -------
    int
        ``2**n_wires * itemsize`` (16 bytes per amplitude for complex128, 8 for complex64).

    Examples
    --------
    >>> state_nbytes(30, "complex64") / 2**30  # GiB
    8.0
    """
    n = validate_n_wires(n_wires)
    return (1 << n) * as_complex_dtype(dtype).itemsize


def _format_bytes(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        size /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


def zero_state(
    n_wires: int,
    dtype: DTypeLike = np.complex128,
    *,
    max_bytes: int | None = None,
) -> StateVector:
    """Allocate the ``|0…0⟩`` state of ``n_wires`` qubits.

    Parameters
    ----------
    n_wires
        Number of qubits, between 1 and :data:`MAX_QUBITS`.
    dtype
        Precision: ``complex128`` (default) or ``complex64``.
    max_bytes
        Optional memory budget. If the state needs more, a :class:`MemoryError` is raised before
        anything is allocated.

    Returns
    -------
    numpy.ndarray
        Contiguous array of shape ``(2**n_wires,)`` with amplitude 1 at index 0 and 0 elsewhere.

    Raises
    ------
    MemoryError
        If the state exceeds ``max_bytes`` or can't be allocated. The message states the qubit
        count, dtype and required memory.

    Examples
    --------
    >>> zero_state(2)
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """
    n = validate_n_wires(n_wires)
    complex_dtype = as_complex_dtype(dtype)
    nbytes = state_nbytes(n, complex_dtype)
    if max_bytes is not None:
        # Runtime check for untyped callers; ``type(...) is int`` also rejects ``bool``.
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}.")
        if nbytes > max_bytes:
            raise MemoryError(
                f"A {n}-qubit {complex_dtype} state needs {_format_bytes(nbytes)}, "
                f"which exceeds max_bytes={_format_bytes(max_bytes)}."
            )
    try:
        state = np.zeros(1 << n, dtype=complex_dtype)
    except (MemoryError, ValueError) as exc:
        raise MemoryError(
            f"Could not allocate a {n}-qubit {complex_dtype} state ({_format_bytes(nbytes)})."
        ) from exc
    state[0] = 1.0
    return state
