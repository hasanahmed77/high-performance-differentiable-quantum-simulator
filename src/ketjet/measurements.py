r"""Reference measurement functions on NumPy statevectors (ADR-0003 D2, D4).

Every function is read-only: the state is never modified.

- :func:`probs`: marginal probabilities, with ``wires[0]`` as the most significant bit of the
  result index (PennyLane's convention).
- :func:`expval`: :math:`\langle O \rangle = \mathrm{Re}\,\langle\psi|O\psi\rangle`.
- :func:`var`: :math:`\mathrm{Var}(O) = \lVert O\psi \rVert^2 - \langle O \rangle^2`, valid for any
  Hermitian :math:`O`.
- :func:`sample`: seeded computational-basis samples (ADR-0003 D5).
- :func:`estimate_expval`: shot-based expectation value via eigenbasis rotation (ADR-0003 D5).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ketjet.gates import GATES
from ketjet.kernels import Wires, apply_matrix, n_wires_of
from ketjet.observables import Hermitian, Observable, Pauli, Prod, Sum
from ketjet.statevector import StateVector

__all__ = [
    "Probabilities",
    "Samples",
    "Seed",
    "estimate_expval",
    "expval",
    "probs",
    "sample",
    "var",
]

Probabilities = npt.NDArray[np.floating[Any]]
"""Real probabilities, ``float64`` for complex128 states and ``float32`` for complex64."""

Samples = npt.NDArray[np.int64]
"""Measured bits, shape ``(shots, n_measured_wires)``, values 0 or 1."""

Seed = int | np.random.Generator | None
"""An integer seed, a NumPy ``Generator`` (used as is), or ``None`` for fresh entropy."""


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


# --------------------------------------------------------------------------- sampling


def _check_shots(shots: int) -> int:
    if type(shots) is not int or shots < 1:  # runtime check for untyped callers
        raise ValueError(f"shots must be a positive int, got {shots!r}.")
    return shots


def sample(
    state: StateVector, shots: int, wires: Sequence[int] | None = None, seed: Seed = None
) -> Samples:
    """Draw computational-basis measurement samples (ADR-0003 D5).

    Parameters
    ----------
    state
        Flat statevector (not modified).
    shots
        Number of samples.
    wires
        Wires to measure (default: all); column ``j`` of the result is ``wires[j]``.
    seed
        ``int`` or ``numpy.random.Generator`` for reproducible samples; ``None`` for fresh entropy.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of 0/1 bits with shape ``(shots, len(wires))``, PennyLane's layout.

    Notes
    -----
    Probabilities are accumulated in float64 and renormalized before drawing, so complex64
    rounding can't bias or break the draw.

    Examples
    --------
    >>> import numpy as np
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2) + 0j
    >>> bits = sample(bell, shots=1000, seed=7)
    >>> bits.shape, bool((bits[:, 0] == bits[:, 1]).all())
    ((1000, 2), True)
    """
    n_shots = _check_shots(shots)
    measured = _measured_wires(wires, n_wires_of(state))
    p = probs(state, measured).astype(np.float64)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(p.size, size=n_shots, p=p)
    k = len(measured)
    bits: Samples = (outcomes[:, None] >> np.arange(k - 1, -1, -1)) & 1
    return bits.astype(np.int64)


_H = GATES["H"].matrix()
_Y_TO_Z = GATES["H"].matrix() @ GATES["Sdg"].matrix()  # S† then H maps the Y eigenbasis to Z


def _eigenvalue_samples(
    state: StateVector, observable: Observable, shots: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Sample eigenvalues of a non-Sum observable by rotating into its eigenbasis."""
    factors = observable.factors if isinstance(observable, Prod) else (observable,)
    rotated = state
    # Per factor: its measured wires and a function from those bits to eigenvalues.
    readouts: list[tuple[tuple[int, ...], Any]] = []
    for factor in factors:
        if isinstance(factor, Pauli):
            for w, letter in factor.word:
                if letter == "X":
                    rotated = apply_matrix(rotated, _H, [w])
                elif letter == "Y":
                    rotated = apply_matrix(rotated, _Y_TO_Z, [w])
            if factor.wires:
                readouts.append((factor.wires, lambda b: 1.0 - 2.0 * (b.sum(axis=1) % 2)))
        elif isinstance(factor, Hermitian):
            eigvals, eigvecs = np.linalg.eigh(factor.base_matrix)
            rotated = apply_matrix(rotated, eigvecs.conj().T, factor.target_wires)
            weights = 1 << np.arange(len(factor.target_wires) - 1, -1, -1)
            readouts.append((factor.target_wires, lambda b, ev=eigvals, wt=weights: ev[b @ wt]))
        else:  # pragma: no cover - Prod factors are Pauli or Hermitian by construction
            raise TypeError(type(factor).__name__)
    if not readouts:  # identity: every shot measures +1
        return np.ones(shots)
    all_wires = tuple(w for wires, _ in readouts for w in wires)
    bits = sample(rotated, shots, all_wires, rng)
    values = np.ones(shots)
    column = 0
    for wires, readout in readouts:
        values = values * readout(bits[:, column : column + len(wires)])
        column += len(wires)
    return values


def estimate_expval(
    state: StateVector, observable: Observable, shots: int, seed: Seed = None
) -> float:
    """Estimate an expectation value from ``shots`` measurements (ADR-0003 D5).

    Each term is rotated into its eigenbasis (X: H; Y: S†·H; ``Hermitian``: its eigenvectors),
    sampled, and its measured eigenvalues averaged. A :class:`~ketjet.observables.Sum` samples
    every term with the full shot budget, like ``default.qubit`` without term grouping.

    Examples
    --------
    >>> import numpy as np
    >>> from ketjet.observables import Z
    >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2) + 0j
    >>> estimate_expval(bell, Z(0) @ Z(1), shots=100, seed=1)
    1.0
    """
    n_shots = _check_shots(shots)
    rng = np.random.default_rng(seed)
    terms = observable.terms if isinstance(observable, Sum) else ((1.0, observable),)
    _check_observable_fits(state, observable)
    return float(
        sum(c * _eigenvalue_samples(state, term, n_shots, rng).mean() for c, term in terms)
    )


def _check_observable_fits(state: StateVector, observable: Observable) -> None:
    n = n_wires_of(state)
    outside = [w for w in observable.wires if w >= n]
    if outside:
        raise ValueError(
            f"{observable!r} acts on wire(s) {outside}, but the state has {n} qubit(s)."
        )
