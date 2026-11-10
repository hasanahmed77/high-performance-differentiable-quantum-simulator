"""Immutable circuit operations and named gate constructors (ADR-0002).

An :class:`Operation` is a base gate from :data:`ketjet.gates.GATES` applied to target wires,
optionally with control wires. Controlled gates are structure, not separate gates: ``CNOT(0, 1)`` is
``Operation("X", wires=(1,), controls=(0,))``. That lets kernels touch only the control-satisfied
subspace and makes any gate controllable.

Operations validate themselves on construction (ADR-0002 D7), so invalid operations never reach a
backend.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from ketjet.gates import GateSpec, Matrix, get_gate
from ketjet.statevector import DTypeLike, as_complex_dtype

__all__ = [
    "CNOT",
    "CRX",
    "CRY",
    "CRZ",
    "CSWAP",
    "CY",
    "CZ",
    "ISWAP",
    "RX",
    "RY",
    "RZ",
    "SWAP",
    "SX",
    "U3",
    "H",
    "I",
    "MultiControlledX",
    "Operation",
    "ParamLike",
    "Parameter",
    "PhaseShift",
    "Rot",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "Toffoli",
    "X",
    "Y",
    "Z",
    "ctrl",
    "display_name",
]


@dataclass(frozen=True, slots=True)
class Parameter:
    """A named symbolic parameter, bound to a value before execution (ADR-0002 D4).

    Examples
    --------
    >>> theta = Parameter("theta")
    >>> RX(theta, 0).is_parametrized_symbolically
    True
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the name."""
        if type(self.name) is not str or not self.name:  # runtime check for untyped callers
            raise ValueError("Parameter name must be a non-empty string.")


ParamLike = float | Parameter
"""A gate parameter: a concrete real number or a symbolic :class:`Parameter`."""


def _check_wire(w: Any, role: str) -> int:
    if isinstance(w, bool) or not isinstance(w, int | np.integer):
        raise TypeError(f"{role} wires must be ints, got {type(w).__name__}.")
    if w < 0:
        raise ValueError(f"{role} wires must be non-negative, got {w}.")
    return int(w)


def _check_param(p: Any, gate: str) -> ParamLike:
    if isinstance(p, Parameter):
        return p
    if isinstance(p, bool) or not isinstance(p, Real):
        raise TypeError(f"{gate} parameters must be real numbers or Parameters, got {p!r}.")
    value = float(p)
    if not np.isfinite(value):
        raise ValueError(f"{gate} parameters must be finite, got {value}.")
    return value


@dataclass(frozen=True, slots=True)
class Operation:
    """A base gate applied to target wires, optionally controlled (ADR-0002 D1, D2).

    Parameters
    ----------
    gate
        Name of a base gate in :data:`ketjet.gates.GATES`.
    wires
        Target wires, in the order matching the gate's matrix.
    params
        Gate parameters: real numbers or :class:`Parameter` symbols.
    controls
        Control wires, disjoint from ``wires``.
    control_values
        Per-control trigger values. Empty means "all controls must be 1".

    Raises
    ------
    KeyError
        If ``gate`` is unknown.
    ValueError, TypeError
        If wires, controls, parameters or control values are invalid.
    """

    gate: str
    wires: tuple[int, ...]
    params: tuple[ParamLike, ...] = ()
    controls: tuple[int, ...] = ()
    control_values: tuple[bool, ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate and normalize every field (ADR-0002 D7)."""
        spec = get_gate(self.gate)
        wires = tuple(_check_wire(w, "Target") for w in self.wires)
        controls = tuple(_check_wire(w, "Control") for w in self.controls)
        if len(wires) != spec.num_wires:
            raise ValueError(f"{self.gate} acts on {spec.num_wires} wire(s), got {len(wires)}.")
        all_wires = controls + wires
        if len(set(all_wires)) != len(all_wires):
            raise ValueError(
                f"Wires must be distinct across targets and controls, got targets {wires} "
                f"and controls {controls}."
            )
        params = tuple(_check_param(p, self.gate) for p in self.params)
        if len(params) != spec.num_params:
            raise ValueError(
                f"{self.gate} takes {spec.num_params} parameter(s), got {len(params)}."
            )
        values = tuple(self.control_values) or (True,) * len(controls)
        if len(values) != len(controls):
            raise ValueError(
                f"Got {len(values)} control value(s) for {len(controls)} control wire(s)."
            )
        if not all(isinstance(v, bool | np.bool_) for v in values):
            raise TypeError("Control values must be booleans.")
        # Normalize (frozen dataclass: assign through object.__setattr__).
        object.__setattr__(self, "wires", wires)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "control_values", tuple(bool(v) for v in values))

    # ------------------------------------------------------------------- properties

    @property
    def spec(self) -> GateSpec:
        """The base gate's specification."""
        return get_gate(self.gate)

    @property
    def all_wires(self) -> tuple[int, ...]:
        """Controls followed by targets: the wire order of :meth:`matrix`."""
        return self.controls + self.wires

    @property
    def is_parametrized_symbolically(self) -> bool:
        """``True`` if any parameter is an unbound :class:`Parameter`."""
        return any(isinstance(p, Parameter) for p in self.params)

    def _numeric_params(self) -> tuple[float, ...]:
        if self.is_parametrized_symbolically:
            unbound = [p.name for p in self.params if isinstance(p, Parameter)]
            raise ValueError(f"{self.gate} has unbound parameter(s): {', '.join(unbound)}.")
        return tuple(float(p) for p in self.params if not isinstance(p, Parameter))

    # ------------------------------------------------------------------- matrices

    def base_matrix(self, dtype: DTypeLike = np.complex128) -> Matrix:
        """Matrix of the base gate on :attr:`wires` (controls ignored)."""
        return self.spec.matrix(self._numeric_params(), dtype)

    def matrix(self, dtype: DTypeLike = np.complex128) -> Matrix:
        """Full matrix on :attr:`all_wires` (controls first, most significant bit first).

        For a controlled gate this is the identity, except on the block where every control
        matches its trigger value, where it's the base matrix.

        Examples
        --------
        >>> CNOT(0, 1).matrix().real.astype(int)
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])
        """
        base = self.base_matrix(dtype)
        n_ctrl = len(self.controls)
        if n_ctrl == 0:
            return base
        k = 1 << len(self.wires)
        full = np.eye(k << n_ctrl, dtype=as_complex_dtype(dtype))
        trigger = 0
        for v in self.control_values:
            trigger = (trigger << 1) | int(v)
        start = trigger * k
        full[start : start + k, start : start + k] = base
        return full

    def bind(self, values: Mapping[str, float]) -> "Operation":
        """Return a copy with symbolic parameters replaced by values from ``values`` (by name).

        Symbols missing from ``values`` stay symbolic.
        """
        params = tuple(
            values.get(p.name, p) if isinstance(p, Parameter) else p for p in self.params
        )
        if params == self.params:
            return self
        return Operation(self.gate, self.wires, params, self.controls, self.control_values)

    def inverse(self) -> "Operation":
        """The inverse operation (same wires and controls)."""
        name, params = self.spec.inverse(self._numeric_params())
        return Operation(name, self.wires, params, self.controls, self.control_values)


# --------------------------------------------------------------------------- constructors


def ctrl(op: Operation, controls: Sequence[int], control_values: Sequence[bool] = ()) -> Operation:
    """Add control wires to an operation (outermost controls first).

    Examples
    --------
    >>> ctrl(X(2), [0, 1]) == Toffoli(0, 1, 2)
    True
    """
    values = tuple(control_values) or (True,) * len(controls)
    return Operation(
        op.gate,
        op.wires,
        op.params,
        tuple(controls) + op.controls,
        values + op.control_values,
    )


def I(wire: int) -> Operation:  # noqa: E743, N802
    """Identity."""
    return Operation("I", (wire,))


def X(wire: int) -> Operation:  # noqa: N802
    """Pauli-X (NOT)."""
    return Operation("X", (wire,))


def Y(wire: int) -> Operation:  # noqa: N802
    """Pauli-Y."""
    return Operation("Y", (wire,))


def Z(wire: int) -> Operation:  # noqa: N802
    """Pauli-Z."""
    return Operation("Z", (wire,))


def H(wire: int) -> Operation:  # noqa: N802
    """Hadamard."""
    return Operation("H", (wire,))


def S(wire: int) -> Operation:  # noqa: N802
    """Phase gate ``diag(1, i)``."""
    return Operation("S", (wire,))


def Sdg(wire: int) -> Operation:  # noqa: N802
    """Inverse phase gate ``diag(1, -i)``."""
    return Operation("Sdg", (wire,))


def T(wire: int) -> Operation:  # noqa: N802
    """T gate ``diag(1, e^{iπ/4})``."""
    return Operation("T", (wire,))


def Tdg(wire: int) -> Operation:  # noqa: N802
    """Inverse T gate."""
    return Operation("Tdg", (wire,))


def SX(wire: int) -> Operation:  # noqa: N802
    """Square root of X."""
    return Operation("SX", (wire,))


def RX(theta: ParamLike, wire: int) -> Operation:  # noqa: N802
    """X rotation ``exp(-iθX/2)``."""
    return Operation("RX", (wire,), (theta,))


def RY(theta: ParamLike, wire: int) -> Operation:  # noqa: N802
    """Y rotation ``exp(-iθY/2)``."""
    return Operation("RY", (wire,), (theta,))


def RZ(theta: ParamLike, wire: int) -> Operation:  # noqa: N802
    """Z rotation ``exp(-iθZ/2)``."""
    return Operation("RZ", (wire,), (theta,))


def PhaseShift(phi: ParamLike, wire: int) -> Operation:  # noqa: N802
    """Phase shift ``diag(1, e^{iφ})``."""
    return Operation("PhaseShift", (wire,), (phi,))


def Rot(phi: ParamLike, theta: ParamLike, omega: ParamLike, wire: int) -> Operation:  # noqa: N802
    """General rotation ``RZ(ω) RY(θ) RZ(φ)``."""
    return Operation("Rot", (wire,), (phi, theta, omega))


def U3(theta: ParamLike, phi: ParamLike, delta: ParamLike, wire: int) -> Operation:  # noqa: N802
    """General single-qubit unitary (PennyLane's ``U3`` convention)."""
    return Operation("U3", (wire,), (theta, phi, delta))


def SWAP(wire0: int, wire1: int) -> Operation:  # noqa: N802
    """Swap two qubits."""
    return Operation("SWAP", (wire0, wire1))


def ISWAP(wire0: int, wire1: int) -> Operation:  # noqa: N802
    """iSWAP: swap with a phase of ``i`` on the swapped amplitudes."""
    return Operation("ISWAP", (wire0, wire1))


def CNOT(control: int, target: int) -> Operation:  # noqa: N802
    """Controlled-X."""
    return Operation("X", (target,), controls=(control,))


def CY(control: int, target: int) -> Operation:  # noqa: N802
    """Controlled-Y."""
    return Operation("Y", (target,), controls=(control,))


def CZ(control: int, target: int) -> Operation:  # noqa: N802
    """Controlled-Z."""
    return Operation("Z", (target,), controls=(control,))


def CRX(theta: ParamLike, control: int, target: int) -> Operation:  # noqa: N802
    """Controlled X rotation."""
    return Operation("RX", (target,), (theta,), (control,))


def CRY(theta: ParamLike, control: int, target: int) -> Operation:  # noqa: N802
    """Controlled Y rotation."""
    return Operation("RY", (target,), (theta,), (control,))


def CRZ(theta: ParamLike, control: int, target: int) -> Operation:  # noqa: N802
    """Controlled Z rotation."""
    return Operation("RZ", (target,), (theta,), (control,))


def Toffoli(control0: int, control1: int, target: int) -> Operation:  # noqa: N802
    """Doubly-controlled X (CCNOT)."""
    return Operation("X", (target,), controls=(control0, control1))


def CSWAP(control: int, wire0: int, wire1: int) -> Operation:  # noqa: N802
    """Controlled SWAP (Fredkin)."""
    return Operation("SWAP", (wire0, wire1), controls=(control,))


def MultiControlledX(  # noqa: N802
    controls: Sequence[int], target: int, control_values: Sequence[bool] = ()
) -> Operation:
    """X on ``target`` controlled by any number of wires, with optional trigger values."""
    return Operation("X", (target,), controls=tuple(controls), control_values=tuple(control_values))


_DISPLAY = {
    ("X", 1): "CNOT",
    ("Y", 1): "CY",
    ("Z", 1): "CZ",
    ("RX", 1): "CRX",
    ("RY", 1): "CRY",
    ("RZ", 1): "CRZ",
    ("X", 2): "Toffoli",
    ("SWAP", 1): "CSWAP",
}


def display_name(op: Operation) -> str:
    """Human-readable name: the textbook name for common controlled gates.

    Examples
    --------
    >>> display_name(CNOT(0, 1)), display_name(ctrl(H(2), [0])), display_name(RX(0.1, 0))
    ('CNOT', 'C-H', 'RX')
    """
    n_ctrl = len(op.controls)
    if n_ctrl == 0:
        return op.gate
    if all(op.control_values) and (op.gate, n_ctrl) in _DISPLAY:
        return _DISPLAY[(op.gate, n_ctrl)]
    if op.gate == "X" and n_ctrl > 2:
        return "MultiControlledX"
    return f"{'C' * n_ctrl}-{op.gate}"
