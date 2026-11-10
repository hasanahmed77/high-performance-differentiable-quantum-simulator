r"""Observables with operator algebra (ADR-0003 D1 to D3).

Observables are immutable values, built with textbook syntax:

>>> from ketjet.observables import X, Z
>>> H = 0.5 * Z(0) @ Z(1) - 0.2 * X(0)
>>> H
Sum(0.5 * Z(0) @ Z(1), -0.2 * X(0))

Every observable can **apply itself to a state**, computing :math:`O|\psi\rangle`. That single
primitive gives expectation values and variances for every observable type (ADR-0003 D2).

Pauli strings never build a matrix: :math:`P|i\rangle = i^{n_Y}(-1)^{\mathrm{popcount}(i \,\&\, z)}
|i \oplus x\rangle`, one gather plus a sign vector (ADR-0003 D3).

Note that ``X``, ``Y`` and ``Z`` here are *observables*, distinct from the gate constructors of the
same name in :mod:`ketjet.operations`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ketjet.kernels import apply_matrix, n_wires_of
from ketjet.statevector import StateVector

__all__ = ["Hermitian", "I", "Observable", "Pauli", "Prod", "Sum", "X", "Y", "Z"]

_PAULI_LETTERS: Final = frozenset("XYZ")
_HERMITIAN_ATOL: Final = 1e-10


def _check_wire(w: Any) -> int:
    if isinstance(w, bool) or not isinstance(w, int | np.integer):
        raise TypeError(f"Wires must be ints, got {type(w).__name__}.")
    if w < 0:
        raise ValueError(f"Wires must be non-negative, got {w}.")
    return int(w)


def _check_coeff(c: Any) -> float:
    if isinstance(c, bool) or not isinstance(c, Real):
        raise TypeError(f"Coefficients must be real numbers, got {c!r}.")
    value = float(c)
    if not np.isfinite(value):
        raise ValueError(f"Coefficients must be finite, got {value}.")
    return value


def _check_in_state(obs: "Observable", n: int) -> None:
    outside = [w for w in obs.wires if w >= n]
    if outside:
        raise ValueError(f"{obs!r} acts on wire(s) {outside}, but the state has {n} qubit(s).")


def _format_coeff(c: float) -> str:
    return f"{c:.6g}"


class Observable(ABC):
    """Base class of all observables: Hermitian operators with algebra.

    Operators: ``a @ b`` (tensor product on disjoint wires), ``c * a`` (real scalar),
    ``a + b``, ``a - b`` and ``-a``.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def wires(self) -> tuple[int, ...]:
        """Wires the observable acts on, sorted."""

    @abstractmethod
    def _apply(self, state: StateVector, n_wires: int) -> StateVector:
        """Return ``O|state⟩`` (wires already validated)."""

    def apply(self, state: StateVector) -> StateVector:
        r"""Return :math:`O|\psi\rangle` as a new array. ``state`` isn't modified.

        Raises
        ------
        ValueError
            If the observable acts on a wire outside the state.
        """
        n = n_wires_of(state)
        _check_in_state(self, n)
        return self._apply(state, n)

    def matrix(self, n_wires: int) -> npt.NDArray[np.complex128]:
        """Dense ``2**n_wires`` matrix (wire 0 = most significant bit). For tests and small n.

        Built column by column from :meth:`apply`, so it's exactly the operator the fast paths
        implement.
        """
        dim = 1 << n_wires
        columns = np.eye(dim, dtype=np.complex128)
        return np.stack([self.apply(columns[:, j]) for j in range(dim)], axis=1)

    # ------------------------------------------------------------------- algebra

    def __matmul__(self, other: "Observable") -> "Observable":
        """Tensor product on disjoint wires."""
        if not isinstance(other, Observable):
            return NotImplemented
        return _tensor(self, other)

    def __mul__(self, scalar: object) -> "Sum":
        """Scale by a real number."""
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            return NotImplemented
        return Sum([(float(scalar), self)])

    __rmul__ = __mul__

    def __neg__(self) -> "Sum":
        """Negate."""
        return Sum([(-1.0, self)])

    def __add__(self, other: "Observable") -> "Sum":
        """Sum of observables."""
        if not isinstance(other, Observable):
            return NotImplemented
        return Sum([(1.0, self), (1.0, other)])

    def __sub__(self, other: "Observable") -> "Sum":
        """Difference of observables."""
        if not isinstance(other, Observable):
            return NotImplemented
        return Sum([(1.0, self), (-1.0, other)])


# --------------------------------------------------------------------------- Pauli strings


class Pauli(Observable):
    """A tensor product of Pauli operators on distinct wires (identity elsewhere).

    Parameters
    ----------
    word
        Mapping or pairs of ``wire -> "X" | "Y" | "Z"``. ``"I"`` entries are dropped. An empty word
        is the identity.

    Examples
    --------
    >>> Pauli({0: "Z", 2: "X"})
    Z(0) @ X(2)
    """

    __slots__ = ("_word",)

    def __init__(self, word: dict[int, str] | Sequence[tuple[int, str]] = ()) -> None:
        items = word.items() if isinstance(word, dict) else word
        checked: dict[int, str] = {}
        for wire, letter in items:
            w = _check_wire(wire)
            if letter not in _PAULI_LETTERS | {"I"}:
                raise ValueError(f"Pauli letters must be I, X, Y or Z, got {letter!r}.")
            if w in checked:
                raise ValueError(f"Wire {w} appears twice in the Pauli word.")
            if letter != "I":
                checked[w] = letter
        self._word: tuple[tuple[int, str], ...] = tuple(sorted(checked.items()))

    @property
    def word(self) -> tuple[tuple[int, str], ...]:
        """``(wire, letter)`` pairs sorted by wire; identities omitted."""
        return self._word

    @property
    def wires(self) -> tuple[int, ...]:
        """Wires carrying a non-identity Pauli."""
        return tuple(w for w, _ in self._word)

    def _masks(self, n: int) -> tuple[int, int, int]:
        x_mask = z_mask = n_y = 0
        for w, letter in self._word:
            bit = 1 << (n - 1 - w)  # wire 0 is the most significant bit (ADR-0001 D2)
            if letter in "XY":
                x_mask |= bit
            if letter in "YZ":
                z_mask |= bit
            n_y += letter == "Y"
        return x_mask, z_mask, n_y

    def _apply(self, state: StateVector, n_wires: int) -> StateVector:
        if not self._word:
            return state.copy()
        x_mask, z_mask, n_y = self._masks(n_wires)
        index = np.arange(1 << n_wires, dtype=np.int64)
        signs = 1 - 2 * (np.bitwise_count(index & z_mask) & 1).astype(np.int8)
        phase = 1j**n_y  # Python scalar: doesn't promote complex64 (NEP 50)
        out = np.empty_like(state)
        out[index ^ x_mask] = state * signs * phase
        return out

    def __eq__(self, other: object) -> bool:
        """Equal when the words are equal."""
        return isinstance(other, Pauli) and self._word == other._word

    def __hash__(self) -> int:
        """Hash of the word."""
        return hash(("Pauli", self._word))

    def __repr__(self) -> str:
        """``Z(0) @ X(2)``, or ``I()`` for the identity."""
        if not self._word:
            return "I()"
        return " @ ".join(f"{letter}({w})" for w, letter in self._word)


def X(wire: int) -> Pauli:  # noqa: N802
    """Pauli-X observable on ``wire``."""
    return Pauli({wire: "X"})


def Y(wire: int) -> Pauli:  # noqa: N802
    """Pauli-Y observable on ``wire``."""
    return Pauli({wire: "Y"})


def Z(wire: int) -> Pauli:  # noqa: N802
    """Pauli-Z observable on ``wire``."""
    return Pauli({wire: "Z"})


def I() -> Pauli:  # noqa: E743, N802
    """Identity observable (expectation value 1)."""
    return Pauli()


# --------------------------------------------------------------------------- Hermitian


class Hermitian(Observable):
    """An arbitrary Hermitian matrix acting on ``wires``.

    Parameters
    ----------
    matrix
        ``2**k x 2**k`` Hermitian matrix; ``wires[0]`` is its most significant bit.
    wires
        The ``k`` distinct wires.

    Raises
    ------
    ValueError
        If the matrix has the wrong shape or isn't Hermitian (to within 1e-10).
    """

    __slots__ = ("_matrix", "_wires")

    def __init__(self, matrix: npt.ArrayLike, wires: Sequence[int]) -> None:
        targets = tuple(_check_wire(w) for w in wires)
        if not targets:
            raise ValueError("Hermitian needs at least one wire.")
        if len(set(targets)) != len(targets):
            raise ValueError(f"Wires must be distinct, got {targets}.")
        m = np.array(matrix, dtype=np.complex128)
        dim = 1 << len(targets)
        if m.shape != (dim, dim):
            raise ValueError(
                f"A Hermitian on {len(targets)} wire(s) needs a {dim}x{dim} matrix, got {m.shape}."
            )
        if not np.allclose(m, m.conj().T, atol=_HERMITIAN_ATOL, rtol=0):
            raise ValueError("Matrix is not Hermitian.")
        m.setflags(write=False)
        self._matrix = m
        self._wires = targets

    @property
    def wires(self) -> tuple[int, ...]:
        """Wires, sorted."""
        return tuple(sorted(self._wires))

    @property
    def target_wires(self) -> tuple[int, ...]:
        """Wires in the order matching the matrix."""
        return self._wires

    @property
    def base_matrix(self) -> npt.NDArray[np.complex128]:
        """The (read-only) matrix, in :attr:`target_wires` order."""
        return self._matrix

    def _apply(self, state: StateVector, n_wires: int) -> StateVector:
        return apply_matrix(state, self._matrix, self._wires)

    def __eq__(self, other: object) -> bool:
        """Equal when wires and matrix are identical."""
        return (
            isinstance(other, Hermitian)
            and self._wires == other._wires
            and np.array_equal(self._matrix, other._matrix)
        )

    def __hash__(self) -> int:
        """Hash of wires and matrix bytes."""
        return hash(("Hermitian", self._wires, self._matrix.tobytes()))

    def __repr__(self) -> str:
        """``Hermitian(<4x4>, wires=(0, 1))``."""
        d = self._matrix.shape[0]
        return f"Hermitian(<{d}x{d}>, wires={self._wires})"


# --------------------------------------------------------------------------- products and sums


class Prod(Observable):
    """Tensor product of observables on disjoint wires.

    Use ``a @ b`` rather than constructing this directly; Pauli factors are merged automatically.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: Sequence[Observable]) -> None:
        flat: list[Observable] = []
        for f in factors:
            if isinstance(f, Sum):
                raise TypeError("Use `@` to take tensor products involving sums.")
            flat.extend(f._factors if isinstance(f, Prod) else [f])
        seen: set[int] = set()
        for f in flat:
            overlap = seen.intersection(f.wires)
            if overlap:
                raise ValueError(f"Tensor product factors overlap on wire(s) {sorted(overlap)}.")
            seen.update(f.wires)
        self._factors: tuple[Observable, ...] = tuple(flat)

    @property
    def factors(self) -> tuple[Observable, ...]:
        """The factors."""
        return self._factors

    @property
    def wires(self) -> tuple[int, ...]:
        """All wires of all factors, sorted."""
        return tuple(sorted(w for f in self._factors for w in f.wires))

    def _apply(self, state: StateVector, n_wires: int) -> StateVector:
        # Factors act on disjoint wires, so they commute: apply them in sequence.
        out = state
        for f in self._factors:
            out = f._apply(out, n_wires)
        return out

    def __eq__(self, other: object) -> bool:
        """Equal when the factors are equal, in order."""
        return isinstance(other, Prod) and self._factors == other._factors

    def __hash__(self) -> int:
        """Hash of the factors."""
        return hash(("Prod", self._factors))

    def __repr__(self) -> str:
        """``Hermitian(...) @ X(1)``."""
        return " @ ".join(repr(f) for f in self._factors)


class Sum(Observable):
    """Real-weighted sum of observables: a Hamiltonian.

    Use ``c * a + b`` rather than constructing this directly. Nested sums are flattened.

    Parameters
    ----------
    terms
        ``(coefficient, observable)`` pairs with real, finite coefficients.
    """

    __slots__ = ("_terms",)

    def __init__(self, terms: Sequence[tuple[float, Observable]]) -> None:
        flat: list[tuple[float, Observable]] = []
        for coeff, obs in terms:
            c = _check_coeff(coeff)
            if not isinstance(obs, Observable):
                raise TypeError(f"Sum terms must be observables, got {type(obs).__name__}.")
            if isinstance(obs, Sum):
                flat.extend((c * inner_c, inner) for inner_c, inner in obs._terms)
            else:
                flat.append((c, obs))
        if not flat:
            raise ValueError("A Sum needs at least one term.")
        self._terms: tuple[tuple[float, Observable], ...] = tuple(flat)

    @property
    def terms(self) -> tuple[tuple[float, Observable], ...]:
        """``(coefficient, observable)`` pairs; no term is itself a :class:`Sum`."""
        return self._terms

    @property
    def wires(self) -> tuple[int, ...]:
        """Union of the terms' wires, sorted."""
        return tuple(sorted({w for _, obs in self._terms for w in obs.wires}))

    def _apply(self, state: StateVector, n_wires: int) -> StateVector:
        out = np.zeros_like(state)
        for coeff, obs in self._terms:
            out += coeff * obs._apply(state, n_wires)
        return out

    def __eq__(self, other: object) -> bool:
        """Equal when terms are equal, in order (no algebraic simplification)."""
        return isinstance(other, Sum) and self._terms == other._terms

    def __hash__(self) -> int:
        """Hash of the terms."""
        return hash(("Sum", self._terms))

    def __repr__(self) -> str:
        """``Sum(0.5 * Z(0) @ Z(1), -0.2 * X(0))``."""
        return "Sum(" + ", ".join(f"{_format_coeff(c)} * {o!r}" for c, o in self._terms) + ")"


def _tensor(a: Observable, b: Observable) -> Observable:
    # Distribute over sums: (Σ cᵢ Aᵢ) @ B = Σ cᵢ (Aᵢ @ B).
    if isinstance(a, Sum):
        return Sum([(c, _tensor(t, b)) for c, t in a.terms])
    if isinstance(b, Sum):
        return Sum([(c, _tensor(a, t)) for c, t in b.terms])
    product = Prod([a, b])  # validates disjoint wires
    paulis = [f for f in product.factors if isinstance(f, Pauli)]
    others = [f for f in product.factors if not isinstance(f, Pauli)]
    merged = Pauli([pair for p in paulis for pair in p.word]) if paulis else None
    if not others:
        assert merged is not None
        return merged
    return Prod(([merged] if merged is not None and merged.word else []) + others)
