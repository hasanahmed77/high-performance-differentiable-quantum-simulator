"""Gate specifications and the standard gate registry (ADR-0002 D3).

A :class:`GateSpec` describes a **base gate**: its arity, its matrix as a function of its
parameters, and structural properties that later enable fast kernels and gate fusion.
Controlled gates are not separate specs: they are a base gate plus control wires (ADR-0002 D2),
built in :mod:`ketjet.operations`.

Matrix conventions match PennyLane: for a gate on wires ``(w0, w1, …)``, ``w0`` is the most
significant bit of the matrix index, and rotations are ``R_P(θ) = exp(-i θ P / 2)``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ketjet.statevector import DTypeLike, as_complex_dtype

__all__ = ["GATES", "GateSpec", "Matrix", "get_gate"]

Matrix = npt.NDArray[np.complexfloating[Any, Any]]
"""A dense complex matrix."""

_MatrixFn = Callable[[tuple[float, ...]], npt.NDArray[np.complex128]]
_InverseFn = Callable[[tuple[float, ...]], tuple[str, tuple[float, ...]]]


@dataclass(frozen=True, slots=True)
class GateSpec:
    """Immutable description of a base gate.

    Attributes
    ----------
    name
        Registry key, e.g. ``"RX"``.
    num_wires
        Number of target wires.
    num_params
        Number of real parameters.
    matrix_fn
        Builds the ``2**num_wires`` square unitary (complex128) from the parameters.
    inverse_fn
        Maps parameters to the ``(gate name, parameters)`` of the inverse gate.
    is_diagonal
        The matrix is diagonal for every parameter value.
    is_permutation
        The matrix is a *phased permutation*: exactly one non-zero entry, of modulus 1, in every
        row and column (e.g. X, Y, SWAP, iSWAP). Such gates move amplitudes without mixing them.
    is_self_inverse
        The gate is its own inverse for every parameter value.
    """

    name: str
    num_wires: int
    num_params: int
    matrix_fn: _MatrixFn
    inverse_fn: _InverseFn
    is_diagonal: bool = False
    is_permutation: bool = False
    is_self_inverse: bool = False

    def matrix(self, params: tuple[float, ...] = (), dtype: DTypeLike = np.complex128) -> Matrix:
        """Return the gate's matrix for concrete ``params`` in the requested precision.

        Raises
        ------
        ValueError
            If the number of parameters is wrong.
        """
        if len(params) != self.num_params:
            raise ValueError(
                f"{self.name} takes {self.num_params} parameter(s), got {len(params)}."
            )
        return self.matrix_fn(params).astype(as_complex_dtype(dtype), copy=False)

    def inverse(self, params: tuple[float, ...] = ()) -> tuple[str, tuple[float, ...]]:
        """Return ``(gate name, parameters)`` of the inverse gate."""
        return self.inverse_fn(params)


# --------------------------------------------------------------------------- matrix builders


def _const(rows: list[list[complex]]) -> _MatrixFn:
    m = np.array(rows, dtype=np.complex128)
    m.setflags(write=False)
    return lambda _params: m


def _named(name: str) -> _InverseFn:
    return lambda _params: (name, ())


def _negate(name: str) -> _InverseFn:
    return lambda params: (name, tuple(-p for p in params))


def _rx(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    c, s = np.cos(params[0] / 2), np.sin(params[0] / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    c, s = np.cos(params[0] / 2), np.sin(params[0] / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    p = np.exp(-0.5j * params[0])
    return np.array([[p, 0], [0, np.conj(p)]], dtype=np.complex128)


def _phase(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * params[0])]], dtype=np.complex128)


def _rot(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    # Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
    phi, theta, omega = params
    return _rz((omega,)) @ _ry((theta,)) @ _rz((phi,))


def _u3(params: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    theta, phi, delta = params
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * delta) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + delta)) * c],
        ],
        dtype=np.complex128,
    )


_SQRT1_2 = 1 / np.sqrt(2)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_T = np.exp(0.25j * np.pi)

_SPECS: tuple[GateSpec, ...] = (
    # --- fixed single-qubit gates
    GateSpec("I", 1, 0, _const([[1, 0], [0, 1]]), _named("I"),
             is_diagonal=True, is_permutation=True, is_self_inverse=True),
    GateSpec("X", 1, 0, _const([[0, 1], [1, 0]]), _named("X"),
             is_permutation=True, is_self_inverse=True),
    GateSpec("Y", 1, 0, _const([[0, -1j], [1j, 0]]), _named("Y"),
             is_permutation=True, is_self_inverse=True),
    GateSpec("Z", 1, 0, _const([[1, 0], [0, -1]]), _named("Z"),
             is_diagonal=True, is_permutation=True, is_self_inverse=True),
    GateSpec("H", 1, 0, _const([[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]]), _named("H"),
             is_self_inverse=True),
    GateSpec("S", 1, 0, _const([[1, 0], [0, 1j]]), _named("Sdg"),
             is_diagonal=True, is_permutation=True),
    GateSpec("Sdg", 1, 0, _const([[1, 0], [0, -1j]]), _named("S"),
             is_diagonal=True, is_permutation=True),
    GateSpec("T", 1, 0, _const([[1, 0], [0, _T]]), _named("Tdg"),
             is_diagonal=True, is_permutation=True),
    GateSpec("Tdg", 1, 0, _const([[1, 0], [0, np.conj(_T)]]), _named("T"),
             is_diagonal=True, is_permutation=True),
    GateSpec("SX", 1, 0, _const(_SX.tolist()), _named("SXdg")),
    GateSpec("SXdg", 1, 0, _const(_SX.conj().T.tolist()), _named("SX")),
    # --- parametric single-qubit gates
    GateSpec("RX", 1, 1, _rx, _negate("RX")),
    GateSpec("RY", 1, 1, _ry, _negate("RY")),
    GateSpec("RZ", 1, 1, _rz, _negate("RZ"), is_diagonal=True),
    GateSpec("PhaseShift", 1, 1, _phase, _negate("PhaseShift"), is_diagonal=True),
    GateSpec("Rot", 1, 3, _rot, lambda p: ("Rot", (-p[2], -p[1], -p[0]))),
    GateSpec("U3", 1, 3, _u3, lambda p: ("U3", (-p[0], -p[2], -p[1]))),
    # --- two-qubit gates (controlled gates are built from base + controls, not listed here)
    GateSpec("SWAP", 2, 0,
             _const([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), _named("SWAP"),
             is_permutation=True, is_self_inverse=True),
    GateSpec("ISWAP", 2, 0,
             _const([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]), _named("ISWAPdg"),
             is_permutation=True),
    GateSpec("ISWAPdg", 2, 0,
             _const([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]), _named("ISWAP"),
             is_permutation=True),
)  # fmt: skip

GATES: Final[Mapping[str, GateSpec]] = MappingProxyType({spec.name: spec for spec in _SPECS})
"""Read-only registry of all base gates, keyed by name."""


def get_gate(name: str) -> GateSpec:
    """Look up a gate specification by name.

    Raises
    ------
    KeyError
        If no gate of that name exists; the message lists the available gates.
    """
    try:
        return GATES[name]
    except KeyError:
        raise KeyError(f"Unknown gate {name!r}. Available: {', '.join(sorted(GATES))}.") from None
