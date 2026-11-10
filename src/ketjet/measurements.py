r"""Reference measurement functions on NumPy statevectors (ADR-0003 D2, D4).

Every function is read-only: the state is never modified.

- :func:`probs`: marginal probabilities, with ``wires[0]`` as the most significant bit of the
  result index (PennyLane's convention).
- :func:`expval`: :math:`\langle O \rangle = \mathrm{Re}\,\langle\psi|O\psi\rangle`.
- :func:`var`: :math:`\mathrm{Var}(O) = \lVert O\psi \rVert^2 - \langle O \rangle^2`, valid for any
  Hermitian :math:`O`.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ketjet.kernels import Wires, n_wires_of
from ketjet.observables import Observable
from ketjet.statevector import StateVector

__all__ = ["Probabilities", "expval", "probs", "var"]

Probabilities = npt.NDArray[np.floating[Any]]
"""Real probabilities, ``float64`` for complex128 states and ``float32`` for complex64."""


def _measured_wires(wires: Wires | None, n: int) -> tuple[int, ...]:
    if wires is None:
        return tuple(range(n))
    checked: list[int] = []
    for w in wires:
        if isinstance(w, bool) or not isinstance(w, int | np.integer):
            raise TypeError(f"Wires must be ints, got {type(w).__name__}.")
        if not 0 <= w < n:
            raise ValueError(f"Wire {w} is out of range for a {n}-qubit state.")
        checked.append(int(w))
    if not checked:
        raise ValueError("At least one wire is required.")
    if len(set(checked)) != len(checked):
        raise ValueError(f"Wires must be distinct, got {tuple(checked)}.")
    return tuple(checked)


def probs(state: StateVector, wires: Sequence[int] | None = None) -> Probabilities:
    """Return the measurement probabilities of ``wires`` (all wires by default).

    Parameters
    ----------
    state
        Flat statevector.
    wires
        Wires to measure, in the order defining the result index: ``wires[0]`` is the most
        significant bit. Other wires are marginalized.

    Returns
    -------
    numpy.ndarray
        ``2**len(wires)`` probabilities, in the state's real precision.

    Examples
    --------
    >>> import numpy as np
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2) + 0j
    >>> probs(bell).round(3), probs(bell, wires=[1]).round(3)
    (array([0.5, 0. , 0. , 0.5]), array([0.5, 0.5]))
    """
    n = n_wires_of(state)
    measured = _measured_wires(wires, n)
    p = (state.real**2 + state.imag**2).reshape((2,) * n)  # |ψ|², keeps the real precision
    others = tuple(w for w in range(n) if w not in measured)
    marginal = p.sum(axis=others) if others else p
    # After summing, remaining axes are the measured wires in ascending order; reorder them.
    order = [sorted(measured).index(w) for w in measured]
    result: Probabilities = np.ascontiguousarray(np.transpose(marginal, order)).reshape(-1)
    return result


def expval(state: StateVector, observable: Observable) -> float:
    r"""Return :math:`\langle\psi|O|\psi\rangle` (real, since :math:`O` is Hermitian).

    Examples
    --------
    >>> import numpy as np
    >>> from ketjet.observables import X, Z
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2) + 0j
    >>> round(expval(bell, Z(0) @ Z(1)), 12), round(expval(bell, X(0)), 12)
    (1.0, 0.0)
    """
    return float(np.vdot(state, observable.apply(state)).real)


def var(state: StateVector, observable: Observable) -> float:
    r"""Return :math:`\lVert O\psi \rVert^2 - \langle O \rangle^2`.

    Clipped at 0, since rounding can make an exact eigenstate's variance slightly negative.

    Examples
    --------
    >>> import numpy as np
    >>> from ketjet.observables import X, Z
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2) + 0j
    >>> round(var(bell, Z(0) @ Z(1)), 12), round(var(bell, X(0)), 12)
    (0.0, 1.0)
    """
    applied = observable.apply(state)
    mean = float(np.vdot(state, applied).real)
    second_moment = float(np.vdot(applied, applied).real)
    return max(second_moment - mean * mean, 0.0)
