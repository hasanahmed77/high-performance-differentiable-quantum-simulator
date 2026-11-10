"""The user-facing circuit builder (ADR-0002 D6).

A :class:`Circuit` is a mutable, append-only builder with fluent gate methods. It stores
immutable :class:`~ketjet.operations.Operation` values, and :attr:`Circuit.operations` returns a
tuple snapshot, so compiler passes and backends can never be affected by later appends.

Examples
--------
>>> from ketjet import Circuit
>>> bell = Circuit(2).h(0).cnot(0, 1)
>>> bell
Circuit(2) [H(0), CNOT(0, 1)]
>>> len(bell), bell.n_wires
(2, 2)
"""

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Self

from ketjet import measurements
from ketjet import operations as ops
from ketjet.backends.base import Backend
from ketjet.backends.numpy_backend import NumPyBackend
from ketjet.measurements import Probabilities, Samples, Seed
from ketjet.observables import Observable
from ketjet.operations import Operation, Parameter, ParamLike, display_name
from ketjet.statevector import DTypeLike, StateVector, validate_n_wires

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ketjet import gradients

__all__ = ["Circuit", "format_operation"]

_FORMAT: Final = "ketjet.circuit"
_VERSION: Final = 1


def _format_param(p: ParamLike) -> str:
    return p.name if isinstance(p, Parameter) else f"{p:.6g}"


def format_operation(op: Operation) -> str:
    """Format an operation the way its constructor is called.

    Examples
    --------
    >>> format_operation(ops.CRX(0.25, 0, 1)), format_operation(ops.ctrl(ops.H(2), [0]))
    ('CRX(0.25, 0, 1)', 'ctrl(H(2), [0])')
    """
    name = display_name(op)
    params = [_format_param(p) for p in op.params]
    if not op.controls or ("-" not in name and name != "MultiControlledX"):
        args = params + [str(w) for w in (op.controls + op.wires)]
        return f"{name}({', '.join(args)})"
    if name == "MultiControlledX":
        return f"MultiControlledX({list(op.controls)}, {op.wires[0]}, {list(op.control_values)})"
    base = format_operation(Operation(op.gate, op.wires, op.params))
    if all(op.control_values):
        return f"ctrl({base}, {list(op.controls)})"
    return f"ctrl({base}, {list(op.controls)}, {list(op.control_values)})"


class Circuit:
    """A quantum circuit on ``n_wires`` qubits, built by appending operations.

    Parameters
    ----------
    n_wires
        Number of qubits (wires ``0 … n_wires-1``).
    operations
        Optional initial operations, validated as if appended one by one.

    Raises
    ------
    ValueError
        If an operation uses a wire outside the circuit.
    """

    __slots__ = ("_n_wires", "_ops")
    __hash__ = None  # type: ignore[assignment]  # mutable, so unhashable

    def __init__(self, n_wires: int, operations: Iterable[Operation] = ()) -> None:
        self._n_wires = validate_n_wires(n_wires)
        self._ops: list[Operation] = []
        self.extend(operations)

    # ------------------------------------------------------------------- core

    @property
    def n_wires(self) -> int:
        """Number of qubits."""
        return self._n_wires

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Immutable snapshot of the operations, in order."""
        return tuple(self._ops)

    def append(self, op: Operation) -> Self:
        """Append one operation and return the circuit, for chaining.

        Raises
        ------
        TypeError
            If ``op`` isn't an :class:`~ketjet.operations.Operation`.
        ValueError
            If ``op`` uses a wire outside ``0 … n_wires-1``.
        """
        if not isinstance(op, Operation):
            raise TypeError(f"Expected an Operation, got {type(op).__name__}.")
        out_of_range = [w for w in op.all_wires if w >= self._n_wires]
        if out_of_range:
            raise ValueError(
                f"{format_operation(op)} uses wire(s) {out_of_range}, but the circuit has "
                f"{self._n_wires} wire(s)."
            )
        self._ops.append(op)
        return self

    def extend(self, operations: Iterable[Operation]) -> Self:
        """Append several operations in order and return the circuit."""
        for op in operations:
            self.append(op)
        return self

    def copy(self) -> "Circuit":
        """Return an independent copy (operations are immutable, so this is cheap)."""
        return Circuit(self._n_wires, self._ops)

    # ------------------------------------------------------------------- parameters

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Unique symbolic parameters, in order of first appearance."""
        seen: dict[Parameter, None] = {}
        for op in self._ops:
            for p in op.params:
                if isinstance(p, Parameter):
                    seen.setdefault(p, None)
        return tuple(seen)

    @property
    def parameter_slots(self) -> tuple[tuple[int, int, Parameter], ...]:
        """Every symbolic parameter position as ``(operation index, parameter index, symbol)``.

        These are the trainable slots a gradient is taken with respect to (QS-402, QS-404). A
        symbol used in several gates appears once per slot.

        Examples
        --------
        >>> theta = Parameter("theta")
        >>> Circuit(2).rx(theta, 0).cnot(0, 1).crz(theta, 0, 1).parameter_slots
        ((0, 0, Parameter(name='theta')), (2, 0, Parameter(name='theta')))
        """
        return tuple(
            (i, j, p)
            for i, op in enumerate(self._ops)
            for j, p in enumerate(op.params)
            if isinstance(p, Parameter)
        )

    @property
    def is_bound(self) -> bool:
        """``True`` if no symbolic parameters remain."""
        return not any(op.is_parametrized_symbolically for op in self._ops)

    def bind(
        self,
        values: Mapping[Parameter, float] | Mapping[str, float] | None = None,
        /,
        **kwargs: float,
    ) -> "Circuit":
        """Return a new circuit with symbolic parameters replaced by values.

        Parameters
        ----------
        values
            Mapping from :class:`~ketjet.operations.Parameter` (or its name) to a value.
        **kwargs
            Values by parameter name, e.g. ``bind(theta=0.5)``.

        Returns
        -------
        Circuit
            A new circuit; this one is unchanged. Parameters not given stay symbolic.

        Raises
        ------
        ValueError
            If a name doesn't match any parameter of the circuit, or a value isn't finite.
        TypeError
            If a value isn't a real number.

        Examples
        --------
        >>> theta = Parameter("theta")
        >>> c = Circuit(1).rx(theta, 0)
        >>> c.bind(theta=0.5) == c.bind({theta: 0.5}) == Circuit(1).rx(0.5, 0)
        True
        """
        by_name: dict[str, float] = {
            key.name if isinstance(key, Parameter) else key: value
            for key, value in (values or {}).items()
        }
        by_name.update(kwargs)
        known = {p.name for p in self.parameters}
        unknown = sorted(set(by_name) - known)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {', '.join(unknown)}; the circuit has "
                f"{', '.join(sorted(known)) or 'none'}."
            )
        return Circuit(self._n_wires, (op.bind(by_name) for op in self._ops))

    def inverse(self) -> "Circuit":
        """Return the inverse circuit: inverted operations in reverse order."""
        return Circuit(self._n_wires, (op.inverse() for op in reversed(self._ops)))

    def __len__(self) -> int:
        """Return the number of operations."""
        return len(self._ops)

    def __iter__(self) -> Iterator[Operation]:
        """Iterate over the operations in order."""
        return iter(tuple(self._ops))

    def __eq__(self, other: object) -> bool:
        """Circuits are equal when they have the same wire count and operations."""
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._n_wires == other._n_wires and self._ops == other._ops

    def __repr__(self) -> str:
        """Compact one-line representation."""
        body = ", ".join(format_operation(op) for op in self._ops)
        return f"Circuit({self._n_wires}) [{body}]"

    # ------------------------------------------------------------------- execution

    def _final_state(
        self, backend: Backend[Any] | None, dtype: DTypeLike
    ) -> tuple[Any, Backend[Any]]:
        if not self.is_bound:
            names = ", ".join(p.name for p in self.parameters)
            raise ValueError(
                f"The circuit has unbound parameter(s): {names}. Call bind(...) before running it."
            )
        engine: Backend[Any] = backend if backend is not None else NumPyBackend()
        state = engine.apply_operations(engine.zero_state(self._n_wires, dtype), self._ops)
        return state, engine

    def state(
        self, *, backend: Backend[Any] | None = None, dtype: DTypeLike = "complex128"
    ) -> StateVector:
        """Run the circuit from ``|0…0⟩`` and return the final amplitudes.

        Parameters
        ----------
        backend
            Backend to run on (default: :class:`~ketjet.backends.NumPyBackend`).
        dtype
            Precision: ``"complex128"`` (default) or ``"complex64"``.

        Raises
        ------
        ValueError
            If symbolic parameters are unbound.
        """
        final, engine = self._final_state(backend, dtype)
        return engine.to_numpy(final)

    def probs(
        self,
        wires: Sequence[int] | None = None,
        *,
        backend: Backend[Any] | None = None,
        dtype: DTypeLike = "complex128",
    ) -> Probabilities:
        """Run the circuit and return measurement probabilities of ``wires`` (default: all).

        ``wires[0]`` is the most significant bit of the result index.

        Examples
        --------
        >>> Circuit(2).h(0).cnot(0, 1).probs().round(3)
        array([0.5, 0. , 0. , 0.5])
        """
        final, engine = self._final_state(backend, dtype)
        return engine.probs(final, wires)

    def expval(
        self,
        observable: Observable,
        *,
        shots: int | None = None,
        seed: Seed = None,
        backend: Backend[Any] | None = None,
        dtype: DTypeLike = "complex128",
    ) -> float:
        """Run the circuit and return the expectation value of ``observable``.

        Parameters
        ----------
        observable
            The observable.
        shots
            ``None`` (default) for the exact value; otherwise the number of measurements used to
            estimate it (ADR-0003 D5).
        seed
            Seed for shot-based estimates.
        backend, dtype
            As for :meth:`state`.

        Examples
        --------
        >>> from ketjet.observables import Z
        >>> round(Circuit(2).h(0).cnot(0, 1).expval(Z(0) @ Z(1)), 12)
        1.0
        """
        final, engine = self._final_state(backend, dtype)
        if shots is None:
            return engine.expval(final, observable)
        # Eigenbasis rotation needs arbitrary unitaries (QubitUnitary, QS-207), so the estimate
        # currently runs on the host copy of the state. Device-side estimation: QS-903.
        return measurements.estimate_expval(engine.to_numpy(final), observable, shots, seed)

    def sample(
        self,
        shots: int,
        wires: Sequence[int] | None = None,
        *,
        seed: Seed = None,
        backend: Backend[Any] | None = None,
        dtype: DTypeLike = "complex128",
    ) -> Samples:
        """Run the circuit and return ``(shots, len(wires))`` measured bits.

        Examples
        --------
        >>> bits = Circuit(2).h(0).cnot(0, 1).sample(5, seed=7)
        >>> bits.shape, bool((bits[:, 0] == bits[:, 1]).all())
        ((5, 2), True)
        """
        final, engine = self._final_state(backend, dtype)
        return engine.sample(final, shots, wires, seed)

    def var(
        self,
        observable: Observable,
        *,
        backend: Backend[Any] | None = None,
        dtype: DTypeLike = "complex128",
    ) -> float:
        """Run the circuit and return the exact variance of ``observable``."""
        final, engine = self._final_state(backend, dtype)
        return engine.var(final, observable)

    def gradient(
        self,
        observable: Observable,
        values: "gradients.Values",
        *,
        method: "gradients.Method" = "adjoint",
        backend: Backend[Any] | None = None,
        dtype: DTypeLike = "complex128",
    ) -> "npt.NDArray[np.float64]":
        """Return the gradient of ``expval(observable)`` with respect to :attr:`parameters`.

        See :func:`ketjet.gradients.grad`.

        Examples
        --------
        >>> from ketjet.observables import Z
        >>> theta = Parameter("theta")
        >>> Circuit(1).rx(theta, 0).gradient(Z(0), {"theta": 0.0}).round(6)
        array([0.])
        """
        from ketjet import gradients  # noqa: PLC0415  # gradients imports this module

        return gradients.grad(self, observable, values, method=method, backend=backend, dtype=dtype)

    # ------------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (versioned format)."""

        def param(p: ParamLike) -> Any:
            return {"parameter": p.name} if isinstance(p, Parameter) else p

        return {
            "format": _FORMAT,
            "version": _VERSION,
            "n_wires": self._n_wires,
            "operations": [
                {
                    "gate": op.gate,
                    "wires": list(op.wires),
                    "params": [param(p) for p in op.params],
                    "controls": list(op.controls),
                    "control_values": list(op.control_values),
                }
                for op in self._ops
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Circuit":
        """Rebuild a circuit from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If the data isn't a ketjet circuit or uses an unsupported format version.
        """
        if data.get("format") != _FORMAT:
            raise ValueError(f"Not a ketjet circuit (format={data.get('format')!r}).")
        if data.get("version") != _VERSION:
            raise ValueError(f"Unsupported circuit format version {data.get('version')!r}.")

        def param(p: Any) -> ParamLike:
            return Parameter(p["parameter"]) if isinstance(p, dict) else p

        return cls(
            data["n_wires"],
            (
                Operation(
                    o["gate"],
                    tuple(o["wires"]),
                    tuple(param(p) for p in o["params"]),
                    tuple(o["controls"]),
                    tuple(o["control_values"]),
                )
                for o in data["operations"]
            ),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Circuit":
        """Rebuild a circuit from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------- fluent gates

    def i(self, wire: int) -> Self:
        """Append an identity."""
        return self.append(ops.I(wire))

    def x(self, wire: int) -> Self:
        """Append Pauli-X."""
        return self.append(ops.X(wire))

    def y(self, wire: int) -> Self:
        """Append Pauli-Y."""
        return self.append(ops.Y(wire))

    def z(self, wire: int) -> Self:
        """Append Pauli-Z."""
        return self.append(ops.Z(wire))

    def h(self, wire: int) -> Self:
        """Append a Hadamard."""
        return self.append(ops.H(wire))

    def s(self, wire: int) -> Self:
        """Append S."""
        return self.append(ops.S(wire))

    def sdg(self, wire: int) -> Self:
        """Append S†."""
        return self.append(ops.Sdg(wire))

    def t(self, wire: int) -> Self:
        """Append T."""
        return self.append(ops.T(wire))

    def tdg(self, wire: int) -> Self:
        """Append T†."""
        return self.append(ops.Tdg(wire))

    def sx(self, wire: int) -> Self:
        """Append √X."""
        return self.append(ops.SX(wire))

    def rx(self, theta: ParamLike, wire: int) -> Self:
        """Append an X rotation."""
        return self.append(ops.RX(theta, wire))

    def ry(self, theta: ParamLike, wire: int) -> Self:
        """Append a Y rotation."""
        return self.append(ops.RY(theta, wire))

    def rz(self, theta: ParamLike, wire: int) -> Self:
        """Append a Z rotation."""
        return self.append(ops.RZ(theta, wire))

    def phase(self, phi: ParamLike, wire: int) -> Self:
        """Append a phase shift."""
        return self.append(ops.PhaseShift(phi, wire))

    def rot(self, phi: ParamLike, theta: ParamLike, omega: ParamLike, wire: int) -> Self:
        """Append a general rotation ``RZ(ω) RY(θ) RZ(φ)``."""
        return self.append(ops.Rot(phi, theta, omega, wire))

    def u3(self, theta: ParamLike, phi: ParamLike, delta: ParamLike, wire: int) -> Self:
        """Append a U3 gate."""
        return self.append(ops.U3(theta, phi, delta, wire))

    def swap(self, wire0: int, wire1: int) -> Self:
        """Append SWAP."""
        return self.append(ops.SWAP(wire0, wire1))

    def iswap(self, wire0: int, wire1: int) -> Self:
        """Append iSWAP."""
        return self.append(ops.ISWAP(wire0, wire1))

    def cnot(self, control: int, target: int) -> Self:
        """Append CNOT."""
        return self.append(ops.CNOT(control, target))

    def cy(self, control: int, target: int) -> Self:
        """Append controlled-Y."""
        return self.append(ops.CY(control, target))

    def cz(self, control: int, target: int) -> Self:
        """Append controlled-Z."""
        return self.append(ops.CZ(control, target))

    def crx(self, theta: ParamLike, control: int, target: int) -> Self:
        """Append a controlled X rotation."""
        return self.append(ops.CRX(theta, control, target))

    def cry(self, theta: ParamLike, control: int, target: int) -> Self:
        """Append a controlled Y rotation."""
        return self.append(ops.CRY(theta, control, target))

    def crz(self, theta: ParamLike, control: int, target: int) -> Self:
        """Append a controlled Z rotation."""
        return self.append(ops.CRZ(theta, control, target))

    def toffoli(self, control0: int, control1: int, target: int) -> Self:
        """Append a Toffoli (CCNOT)."""
        return self.append(ops.Toffoli(control0, control1, target))

    def cswap(self, control: int, wire0: int, wire1: int) -> Self:
        """Append a controlled SWAP (Fredkin)."""
        return self.append(ops.CSWAP(control, wire0, wire1))

    def mcx(
        self, controls: Sequence[int], target: int, control_values: Sequence[bool] = ()
    ) -> Self:
        """Append a multi-controlled X."""
        return self.append(ops.MultiControlledX(controls, target, control_values))
