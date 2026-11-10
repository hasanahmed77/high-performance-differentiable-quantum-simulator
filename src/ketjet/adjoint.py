r"""Reference adjoint differentiation on NumPy statevectors (ADR-0004 D3; Jones and Gacon, 2020).

For :math:`f = \langle\psi|O|\psi\rangle` with :math:`|\psi\rangle = U_N \cdots U_1 |0\rangle`,

.. math::

    \frac{\partial f}{\partial x_s} = 2\,\operatorname{Re}\,
    \langle \lambda_k | \partial_s U_k | \psi_{k-1} \rangle,
    \qquad \langle \lambda_k | = \langle\psi| O\, U_N \cdots U_{k+1}.

Both :math:`|\psi_{k-1}\rangle` and :math:`|\lambda_k\rangle` come from *undoing* one gate at a time
on the way back, so no intermediate state is stored: memory is :math:`2 + M` statevectors for
:math:`M` observables, plus one scratch state, independent of circuit depth.

Backends implement :meth:`~ketjet.backends.Backend.adjoint_jacobian` (ADR-0004 D4); this module is
the NumPy backend's implementation and the oracle for faster ones.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ketjet.kernels import apply_structured
from ketjet.observables import Observable
from ketjet.operations import Operation
from ketjet.statevector import DTypeLike, StateVector, as_complex_dtype, zero_state

__all__ = ["adjoint_jacobian"]


def _apply(state: StateVector, matrix: npt.NDArray[Any], op: Operation) -> StateVector:
    """Apply ``matrix`` (the gate or its adjoint, which share structure) with the fast kernels."""
    return apply_structured(
        state, matrix, op.all_wires, diagonal=op.is_diagonal, permutation=op.is_permutation
    )


def adjoint_jacobian(
    n_wires: int,
    operations: Sequence[Operation],
    observables: Sequence[Observable],
    trainable: Sequence[tuple[int, int]],
    *,
    dtype: DTypeLike = np.complex128,
) -> npt.NDArray[np.float64]:
    r"""Return :math:`J_{ms} = \partial \langle O_m \rangle / \partial x_s` by one backward pass.

    Parameters
    ----------
    n_wires
        Number of qubits; the circuit starts from :math:`|0\cdots0\rangle`.
    operations
        Bound operations, in circuit order.
    observables
        The observables :math:`O_m`.
    trainable
        ``(operation index, parameter index)`` of every slot to differentiate, in column order.
        A slot may appear more than once.
    dtype
        Precision of the states: ``complex128`` (default) or ``complex64``.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(len(observables), len(trainable))``.

    Raises
    ------
    ValueError
        If a trainable slot names an operation or parameter that doesn't exist.

    Examples
    --------
    >>> from ketjet.observables import Z
    >>> from ketjet.operations import RX
    >>> adjoint_jacobian(1, [RX(0.5, 0)], [Z(0)], [(0, 0)]).round(6)  # d cos(x) = -sin(x)
    array([[-0.479426]])
    """
    dtype = as_complex_dtype(dtype)
    ops = list(operations)
    by_op: dict[int, list[tuple[int, int]]] = {}
    for column, (i, j) in enumerate(trainable):
        if not 0 <= i < len(ops) or not 0 <= j < ops[i].spec.num_params:
            raise ValueError(f"Trainable slot {(i, j)} doesn't exist in the circuit.")
        by_op.setdefault(i, []).append((column, j))
    jac = np.zeros((len(observables), len(trainable)))
    if not by_op:
        return jac

    psi = zero_state(n_wires, dtype)
    for op in ops:
        psi = _apply(psi, op.matrix(dtype), op)
    lambdas = [obs.apply(psi) for obs in observables]

    first = min(by_op)  # gates before the first trainable one don't affect any gradient
    for k in range(len(ops) - 1, first - 1, -1):
        op = ops[k]
        adjoint = op.matrix(dtype).conj().T
        psi = _apply(psi, adjoint, op)  # now |ψ_{k-1}⟩
        for column, j in by_op.get(k, ()):
            # A diagonal gate's derivative is diagonal; a permutation's isn't a permutation.
            mu = apply_structured(
                psi, op.derivative_matrix(j, dtype), op.all_wires, diagonal=op.is_diagonal
            )
            for m, lam in enumerate(lambdas):
                jac[m, column] = 2 * np.vdot(lam, mu).real
            del mu  # free the scratch state before the next slot
        if k > first:
            # One at a time, so each old λ is freed at once: building a new list would hold
            # 2M states instead of M.
            for m in range(len(lambdas)):
                lambdas[m] = _apply(lambdas[m], adjoint, op)
    return jac
