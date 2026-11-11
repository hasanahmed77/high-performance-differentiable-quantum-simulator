"""The PennyLane device ``ketjet.qubit`` (ADR-0005).

Install with ``pip install "ketjet[pennylane]"``, then::

    import pennylane as qml

    dev = qml.device("ketjet.qubit", wires=2)

PennyLane decomposes every circuit into the gates ketjet supports (ADR-0005 D3). Each remaining
PennyLane operation then becomes exactly one ketjet :class:`~ketjet.operations.Operation` with the
same parameters in the same order (D2), and the circuit runs on a ketjet backend. Expectation
values, variances and probabilities are computed by ketjet; other measurements use PennyLane's own
definitions on ketjet's state (D5). ``diff_method="adjoint"`` (also what ``"best"`` picks for exact
expectation values) uses ketjet's adjoint Jacobian and single-pass VJP (D4).

This module imports PennyLane. ``import ketjet`` doesn't import it (ADR-0005 D1).
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    diagonalize_measurements,
    no_sampling,
    null_postprocessing,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
)
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP, ProbabilityMP, VarianceMP
from pennylane.ops.op_math import Adjoint, Controlled
from pennylane.tape import QuantumScript
from pennylane.transforms import broadcast_expand, defer_measurements, split_non_commuting
from pennylane.transforms.core import CompilePipeline
from pennylane.wires import Wires

from ketjet.backends import Backend, NumPyBackend
from ketjet.gates import GATES
from ketjet.kernels import apply_matrix
from ketjet.measurements import sample
from ketjet.observables import Hermitian, Observable, Pauli, Sum
from ketjet.operations import Operation
from ketjet.statevector import DTypeLike, StateVector, as_complex_dtype

__all__ = ["KetjetQubit", "to_ketjet", "to_ketjet_observable"]

# --------------------------------------------------------------------------- translation (D2, D3)

_BASE: Final[dict[str, str]] = {
    "Identity": "I",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Hadamard": "H",
    "S": "S",
    "T": "T",
    "SX": "SX",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "PhaseShift": "PhaseShift",
    "Rot": "Rot",
    "U3": "U3",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
}
"""PennyLane name → ketjet gate, with identical parameters in identical order."""

_ADJOINT: Final[dict[str, str]] = {"S": "Sdg", "T": "Tdg", "SX": "SXdg", "ISWAP": "ISWAPdg"}
"""Adjoints of fixed gates: no parameters, so nothing is rewritten."""


def _base_gate(op: Any) -> str | None:
    """Return the ketjet gate for an uncontrolled PennyLane operation, or ``None``.

    The wire count must match too: PennyLane's ``Identity`` takes any number of wires, ketjet's
    ``I`` one, so a multi-wire ``Identity`` is left to ``decompose``.
    """
    if isinstance(op, Adjoint):
        gate = _ADJOINT.get(op.base.name) if not op.base.parameters else None
    else:
        gate = _BASE.get(op.name)
    if gate is None or len(op.wires) != GATES[gate].num_wires:
        return None
    return gate


def _is_supported(op: Any) -> bool:
    """Return ``True`` if :func:`to_ketjet` translates ``op``: ``decompose``'s stop condition."""
    if isinstance(op, qml.GlobalPhase):
        return True
    if isinstance(op, Controlled):
        base = op.base
        return not isinstance(base, Controlled | qml.GlobalPhase) and _base_gate(base) is not None
    return _base_gate(op) is not None


def to_ketjet(op: Any, wire_index: dict[Any, int]) -> Operation:
    """Translate one supported PennyLane operation into one ketjet :class:`Operation` (D2).

    Parameters
    ----------
    op
        A PennyLane operation in the supported set (ADR-0005 D3), other than ``GlobalPhase``.
    wire_index
        Maps each PennyLane wire label to a ketjet wire index.

    Returns
    -------
    Operation
        The same gate, with the same parameters in the same order.

    Raises
    ------
    ValueError
        If ``op`` isn't in the supported set.
    """
    controls: tuple[int, ...] = ()
    values: tuple[bool, ...] = ()
    base = op
    if isinstance(op, Controlled):
        controls = tuple(wire_index[w] for w in op.control_wires)
        values = tuple(bool(v) for v in op.control_values)
        base = op.base
    gate = _base_gate(base) if not isinstance(base, Controlled) else None
    if gate is None:
        raise ValueError(f"ketjet.qubit can't run {op.name} directly; it should be decomposed.")
    wires = tuple(wire_index[w] for w in base.wires)
    params = tuple(float(p) for p in base.parameters)
    return Operation(gate, wires, params, controls, values)


_PAULI: Final[dict[str, str]] = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}


def to_ketjet_observable(obs: Any, wire_index: dict[Any, int]) -> Observable:
    """Translate a PennyLane observable into a ketjet one (ADR-0005 D5).

    Pauli words use ketjet's matrix-free path; ``Prod``, ``SProd``, ``Sum`` and
    ``LinearCombination`` translate structurally; any other observable with a matrix becomes a
    :class:`~ketjet.observables.Hermitian` on its wires.

    Raises
    ------
    ValueError, TypeError
        If the observable can't be represented, e.g. it has a complex coefficient or no matrix.
    """
    name = obs.name
    if name in _PAULI:
        return Pauli([(wire_index[obs.wires[0]], _PAULI[name])])
    if name == "Identity":
        return Pauli(())
    if isinstance(obs, qml.ops.SProd):
        return _real(obs.scalar) * to_ketjet_observable(obs.base, wire_index)
    if isinstance(obs, qml.ops.Prod):
        factors = [to_ketjet_observable(f, wire_index) for f in obs.operands]
        if _disjoint([f.wires for f in factors]):
            result = factors[0]
            for factor in factors[1:]:
                result = result @ factor
            return result
    elif isinstance(obs, qml.ops.Sum | qml.ops.LinearCombination):
        coeffs, terms = obs.terms()
        return Sum(
            [
                (_real(c), to_ketjet_observable(t, wire_index))
                for c, t in zip(coeffs, terms, strict=True)
            ]
        )
    wires = [wire_index[w] for w in obs.wires]
    return Hermitian(np.asarray(qml.matrix(obs, wire_order=obs.wires)), wires)


def _real(c: Any) -> float:
    value = complex(c)
    if value.imag != 0:
        raise ValueError(f"Observable coefficients must be real, got {value}.")
    return value.real


def _disjoint(wire_sets: Sequence[tuple[int, ...]]) -> bool:
    seen: set[int] = set()
    for wires in wire_sets:
        if seen.intersection(wires):
            return False
        seen.update(wires)
    return True


# --------------------------------------------------------------------------- preprocessing


@qml.transform  # type: ignore[untyped-decorator]
def _when_sampled(tape: QuantumScript, inner: Callable[..., Any]) -> Any:
    """Apply the transform ``inner`` only to tapes with shots: analytic tapes keep observables."""
    if not tape.shots:
        return (tape,), null_postprocessing
    return inner(tape)


def _to_eigenbasis(
    host: StateVector, measurements: Sequence[Any], wire_order: Wires
) -> StateVector:
    """Rotate a host state into the eigenbasis of the measured observables.

    PennyLane's ``process_state`` and ``process_samples`` read an observable's outcomes in its
    eigenbasis (its ``diagonalizing_gates``), which ``default.qubit`` rotates to before measuring.
    Pauli observables were already rotated by ``diagonalize_measurements``; ``Hermitian`` and
    ``Projector`` are left to the device. After ``split_non_commuting`` the observables of one
    tape can be measured together, so each distinct rotation gate is applied once, even when
    several measurements share it (e.g. ``expval(H)`` and ``var(H)``).
    """
    index = {w: i for i, w in enumerate(wire_order)}
    applied: set[int] = set()
    for mp in measurements:
        for gate in mp.obs.diagonalizing_gates() if mp.obs is not None else ():
            if gate.hash in applied:
                continue
            applied.add(gate.hash)
            host = apply_matrix(host, qml.matrix(gate), [index[w] for w in gate.wires])
    return host


def _undecomposable(op: Any) -> bool:
    """Return ``True`` for unsupported operations with a matrix but no decomposition."""
    return not _is_supported(op) and not op.has_decomposition and op.has_matrix


@qml.transform  # type: ignore[untyped-decorator]
def _matrix_fallback(tape: QuantumScript) -> Any:
    """Rewrite operations with a matrix but no decomposition as ``QubitUnitary`` (ADR-0005 D3).

    PennyLane decomposes ``QubitUnitary`` on any number of wires, so ``decompose`` can then
    reach the supported set; e.g. ``DoubleExcitationPlus`` and ``BlockEncode`` have no rule of
    their own. That decomposition can't be differentiated (it uses SciPy), so trainable
    parameters raise instead of silently getting a zero gradient.
    """
    operations = []
    for op in tape.operations:
        if not _undecomposable(op):
            operations.append(op)
            continue
        if any(qml.math.requires_grad(p) for p in op.data):
            raise DeviceError(
                f"{op.name} has no decomposition, so ketjet.qubit runs it from its matrix, which "
                "can't be differentiated. Make its parameters non-trainable, or use a device "
                "that supports it natively, such as default.qubit."
            )
        operations.append(qml.QubitUnitary(qml.matrix(op), wires=op.wires))
    return (tape.copy(operations=operations),), null_postprocessing


_ADJOINT_METHODS: Final[frozenset[str]] = frozenset({"adjoint", "best", "device"})
"""Gradient methods the device answers with its own adjoint derivatives (ADR-0005 D4)."""


def _adjoint_supported(circuit: QuantumScript | None) -> bool:
    """Return ``True`` if ``circuit`` is exact and measures only translatable expectation values."""
    if circuit is None:
        return True
    if circuit.shots or not circuit.measurements:
        return False
    index = {w: i for i, w in enumerate(circuit.wires)}
    for mp in circuit.measurements:
        if not isinstance(mp, ExpectationMP):
            return False
        try:
            to_ketjet_observable(mp.obs, index)
        except (ValueError, TypeError):
            return False
    return True


def _format_jacobian(jac: npt.NDArray[np.float64]) -> Any:
    """Shape an ``(M, T)`` Jacobian the way PennyLane expects (as ``default.qubit`` does)."""
    squeezed = np.squeeze(jac)
    if squeezed.ndim == 0:
        return np.array(squeezed)
    if squeezed.ndim == 1:
        return tuple(np.array(j) for j in squeezed)
    return tuple(tuple(np.array(j) for j in row) for row in squeezed)


# --------------------------------------------------------------------------- the device


@simulator_tracking
@single_tape_support
class KetjetQubit(Device):  # type: ignore[misc]  # PennyLane ships no type hints
    """ketjet as a PennyLane device: ``qml.device("ketjet.qubit", wires=n)`` (ADR-0005).

    Parameters
    ----------
    wires
        Number of wires, or their labels (any hashable values). ``None`` (default) uses the
        circuit's wires.
    shots
        Default number of shots, or ``None`` (default) for exact results. A QNode's own ``shots``
        take precedence.
    seed
        ``int``, ``numpy.random.Generator`` or ``None``. A seeded device gives reproducible
        samples; each circuit draws from its own child seed.
    c_dtype
        Precision: ``"complex128"`` (default) or ``"complex64"``.
    backend
        The ketjet backend (default: :class:`~ketjet.backends.NumPyBackend`).

    Examples
    --------
    >>> import pennylane as qml
    >>> dev = qml.device("ketjet.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def bell():
    ...     qml.Hadamard(0)
    ...     qml.CNOT([0, 1])
    ...     return qml.probs()
    >>> bell().round(3).tolist()
    [0.5, 0.0, 0.0, 0.5]
    """

    name = "ketjet.qubit"

    def __init__(
        self,
        wires: Any = None,
        shots: Any = None,
        seed: int | np.random.Generator | None = None,
        c_dtype: DTypeLike = "complex128",
        backend: Backend[Any] | None = None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)
        self._dtype = as_complex_dtype(c_dtype)
        self._rng = np.random.default_rng(seed)
        self._backend: Backend[Any] = backend if backend is not None else NumPyBackend()

    @property
    def c_dtype(self) -> np.dtype[Any]:
        """The precision of the simulation."""
        return self._dtype

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> Any:
        """Rewrite tapes into what ketjet runs natively (ADR-0005 D2)."""
        program = CompilePipeline()
        program.add_transform(validate_device_wires, wires=self.wires, name=self.name)
        program.add_transform(defer_measurements, allow_postselect=False)
        # Decompose down to supported gates, or to gates with only a matrix (which may appear
        # only mid-decomposition, as BlockEncode inside QSVT does); rewrite those as QubitUnitary;
        # then decompose again, which always ends in the supported set.
        program.add_transform(
            decompose,
            stopping_condition=lambda op: _is_supported(op) or _undecomposable(op),
            skip_initial_state_prep=False,
            name=self.name,
        )
        program.add_transform(_matrix_fallback)
        program.add_transform(
            decompose,
            stopping_condition=_is_supported,
            skip_initial_state_prep=False,
            name=self.name,
        )
        program.add_transform(validate_measurements, name=self.name)
        program.add_transform(broadcast_expand)
        program.add_transform(_when_sampled, inner=split_non_commuting)
        program.add_transform(_when_sampled, inner=diagonalize_measurements)
        if execution_config is not None and execution_config.gradient_method == "adjoint":
            program.add_transform(no_sampling, name=f"adjoint + {self.name}")
            program.add_transform(validate_adjoint_trainable_params)
        return program

    # ------------------------------------------------------------------- derivatives (D4)

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Return ``True`` for first-order adjoint derivatives of exact expectation values.

        ``"best"`` and ``"device"`` mean adjoint here. ``"backprop"`` isn't supported: ketjet's
        NumPy kernels can't be traced by autodiff frameworks (ADR-0005 D4).
        """
        if execution_config is None:
            return True
        if (
            execution_config.gradient_method not in _ADJOINT_METHODS
            or execution_config.derivative_order != 1
        ):
            return False
        return _adjoint_supported(circuit)

    def supports_vjp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Return ``True`` where :meth:`supports_derivatives` does."""
        return self.supports_derivatives(execution_config, circuit)

    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        """Resolve ``"best"`` and ``"device"`` to adjoint where the device supports it."""
        config = super().setup_execution_config(config, circuit)
        if config.gradient_method not in _ADJOINT_METHODS or not self.supports_derivatives(
            config, circuit
        ):
            return config
        updates: dict[str, Any] = {"gradient_method": "adjoint"}
        for option in ("use_device_gradient", "use_device_jacobian_product", "grad_on_execution"):
            if getattr(config, option) is None:
                updates[option] = True
        return dataclasses.replace(config, **updates)

    def compute_derivatives(
        self, circuits: Sequence[QuantumScript], execution_config: ExecutionConfig | None = None
    ) -> tuple[Any, ...]:
        """Return each tape's Jacobian from one backend adjoint call (ADR-0004 D4)."""
        return tuple(_format_jacobian(self._adjoint(tape)) for tape in circuits)

    def execute_and_compute_derivatives(
        self, circuits: Sequence[QuantumScript], execution_config: ExecutionConfig | None = None
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return the results and the Jacobians."""
        return self.execute(circuits, execution_config), self.compute_derivatives(circuits)

    def compute_vjp(
        self,
        circuits: Sequence[QuantumScript],
        cotangents: Sequence[Any],
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[Any, ...]:
        """Return each tape's vector-Jacobian product: one backward pass per cotangent (D6)."""
        return tuple(
            self._vjp(tape, cotangent) for tape, cotangent in zip(circuits, cotangents, strict=True)
        )

    def execute_and_compute_vjp(
        self,
        circuits: Sequence[QuantumScript],
        cotangents: Sequence[Any],
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return the results and the vector-Jacobian products."""
        return self.execute(circuits, execution_config), self.compute_vjp(circuits, cotangents)

    def execute(
        self, circuits: Sequence[QuantumScript], execution_config: ExecutionConfig | None = None
    ) -> tuple[Any, ...]:
        """Run each tape on the backend and return PennyLane-shaped results."""
        children = self._rng.spawn(len(circuits))
        return tuple(self._run(tape, rng) for tape, rng in zip(circuits, children, strict=True))

    # ------------------------------------------------------------------- internals

    def _adjoint_problem(
        self, tape: QuantumScript
    ) -> tuple[int, list[Operation], list[Observable], list[tuple[int, int] | None]]:
        """Translate a tape for the backend; each trainable parameter becomes one slot (D2).

        Trainable parameters with no slot (``None``) get a zero derivative: a ``GlobalPhase``
        doesn't change any expectation value, and observable parameters aren't differentiated by
        adjoint (PennyLane warns about them in ``validate_adjoint_trainable_params``).
        """
        wire_order = self._wire_order(tape)
        index = {w: i for i, w in enumerate(wire_order)}
        operations: list[Operation] = []
        position: list[int | None] = []  # tape operation index -> translated index
        for op in tape.operations:
            if isinstance(op, qml.GlobalPhase):
                position.append(None)
                continue
            position.append(len(operations))
            operations.append(to_ketjet(op, index))
        slots: list[tuple[int, int] | None] = []
        for k in tape.trainable_params:
            info = tape.par_info[k]
            # Parameters after the operations belong to the measured observables.
            translated = position[info["op_idx"]] if info["op_idx"] < len(position) else None
            slots.append(None if translated is None else (translated, info["p_idx"]))
        observables = [to_ketjet_observable(mp.obs, index) for mp in tape.measurements]
        return max(len(wire_order), 1), operations, observables, slots

    def _jacobian(
        self,
        n_wires: int,
        operations: list[Operation],
        observables: list[Observable],
        slots: list[tuple[int, int] | None],
    ) -> npt.NDArray[np.float64]:
        """Return the ``(M, T)`` Jacobian; ``None`` slots give zero columns."""
        jac = np.zeros((len(observables), len(slots)))
        real = [k for k, slot in enumerate(slots) if slot is not None]
        if real and observables:
            trainable = [slot for slot in slots if slot is not None]
            jac[:, real] = self._backend.adjoint_jacobian(
                n_wires, operations, observables, trainable, dtype=self._dtype
            )
        return jac

    def _adjoint(self, tape: QuantumScript) -> npt.NDArray[np.float64]:
        return self._jacobian(*self._adjoint_problem(tape))

    def _vjp(self, tape: QuantumScript, cotangent: Any) -> Any:
        r"""Return the VJP: one backward pass over :math:`\sum_m v_m O_m` per cotangent (D6)."""
        n_wires, operations, observables, slots = self._adjoint_problem(tape)
        v = np.asarray(cotangent, dtype=np.float64)
        if len(observables) == 1:
            v = v[np.newaxis]  # PennyLane drops the measurement axis for one measurement
        batched = v.ndim == 2  # (measurements, batch)
        columns = v if batched else v[:, np.newaxis]
        out = np.zeros((len(slots), columns.shape[1]))
        for b in range(columns.shape[1]):
            weights = columns[:, b]
            if not weights.any():
                continue  # a zero cotangent: no simulation needed
            combined = Sum([(float(c), obs) for c, obs in zip(weights, observables, strict=True)])
            out[:, b] = self._jacobian(n_wires, operations, [combined], slots)[0]
        return out if batched else out[:, 0]

    def _wire_order(self, tape: QuantumScript) -> Wires:
        return self.wires if self.wires is not None else tape.wires

    def _simulate(self, tape: QuantumScript, wire_order: Wires) -> Any:
        """Run the tape's operations and return the final state, on the backend."""
        index = {w: i for i, w in enumerate(wire_order)}
        backend = self._backend
        state = backend.zero_state(max(len(wire_order), 1), self._dtype)
        phase = 1.0 + 0.0j
        for op in tape.operations:
            if isinstance(op, qml.GlobalPhase):
                phase *= np.exp(-1j * float(op.parameters[0]))
                continue
            state = backend.apply_operation(state, to_ketjet(op, index))
        if phase != 1:
            state = state * np.asarray(phase, dtype=self._dtype)
        return state

    def _run(self, tape: QuantumScript, rng: np.random.Generator) -> Any:
        wire_order = self._wire_order(tape)
        state = self._simulate(tape, wire_order)
        if tape.shots:
            return self._sampled(tape, state, wire_order, rng)
        results = tuple(self._analytic(mp, state, wire_order) for mp in tape.measurements)
        return results[0] if len(results) == 1 else results

    def _analytic(self, mp: Any, state: Any, wire_order: Wires) -> Any:
        """Return one exact measurement: native for expval, var and probs; PennyLane's otherwise."""
        index = {w: i for i, w in enumerate(wire_order)}
        backend = self._backend
        if isinstance(mp, ExpectationMP | VarianceMP):
            try:
                obs = to_ketjet_observable(mp.obs, index)
            except (ValueError, TypeError):
                pass  # e.g. a complex coefficient: PennyLane decides what it means
            else:
                measure = backend.expval if isinstance(mp, ExpectationMP) else backend.var
                return np.float64(measure(state, obs))
        if isinstance(mp, ProbabilityMP) and mp.obs is None:
            wires = [index[w] for w in mp.wires] if mp.wires else None
            return np.asarray(backend.probs(state, wires), dtype=np.float64)
        host = _to_eigenbasis(backend.to_numpy(state), [mp], wire_order)
        return mp.process_state(host, wire_order)

    def _sampled(
        self, tape: QuantumScript, state: Any, wire_order: Wires, rng: np.random.Generator
    ) -> Any:
        """Return shot-based results, per shot bin, from one batch of ketjet samples."""
        host = _to_eigenbasis(self._backend.to_numpy(state), tape.measurements, wire_order)
        samples = sample(host, tape.shots.total_shots, seed=rng)
        per_bin = []
        for lower, upper in tape.shots.bins():
            chunk = samples[lower:upper]
            values = tuple(mp.process_samples(chunk, wire_order) for mp in tape.measurements)
            per_bin.append(values[0] if len(values) == 1 else values)
        return tuple(per_bin) if tape.shots.has_partitioned_shots else per_bin[0]
