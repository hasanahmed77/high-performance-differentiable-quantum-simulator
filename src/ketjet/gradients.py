r"""Gradients of circuit expectation values (ADR-0004).

Gradients are taken with respect to a circuit's :class:`~ketjet.operations.Parameter` symbols, in
:attr:`Circuit.parameters <ketjet.circuit.Circuit.parameters>` order. Every method first
differentiates each *slot* (one gate parameter holding a symbol), then sums the slots that share a
symbol, which is the chain rule (ADR-0004 D1).

Methods
-------
``"finite-diff"``
    Central differences, :math:`[f(x+h) - f(x-h)] / 2h`. Approximate: a test oracle and fallback,
    never the default (ADR-0004 D7).

Examples
--------
>>> import numpy as np
>>> from ketjet import Circuit
>>> from ketjet.observables import Z
>>> from ketjet.operations import Parameter
>>> theta = Parameter("theta")
>>> g = grad(Circuit(1).ry(theta, 0), Z(0), {"theta": 0.3}, method="finite-diff")
>>> bool(np.isclose(g[0], -np.sin(0.3)))
True
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from ketjet.backends.base import Backend
from ketjet.backends.numpy_backend import NumPyBackend
from ketjet.circuit import Circuit
from ketjet.observables import Observable
from ketjet.operations import Operation, Parameter
from ketjet.statevector import DTypeLike, as_complex_dtype

__all__ = ["Measurement", "Method", "Values", "grad", "jacobian"]

Method = Literal["finite-diff"]
"""A differentiation method name."""

Measurement = Literal["expval", "var"]
"""The differentiated quantity of each observable."""

Values = Mapping[Parameter, float] | Mapping[str, float]
"""Values for every symbol of a circuit, keyed by :class:`Parameter` or by name."""

Jacobian = npt.NDArray[np.float64]

_METHODS: Final[tuple[str, ...]] = ("finite-diff",)
_MEASUREMENTS: Final[tuple[str, ...]] = ("expval", "var")
_DEFAULT_STEP: Final[dict[np.dtype[Any], float]] = {
    np.dtype(np.complex128): 1e-7,  # plan.md §6.3
    np.dtype(np.complex64): 1e-3,  # a smaller step is swamped by float32 rounding
}


@dataclasses.dataclass(frozen=True, slots=True)
class _Problem:
    """A bound circuit with its trainable slots (ADR-0004 D1)."""

    n_wires: int
    operations: tuple[Operation, ...]
    slots: tuple[tuple[int, int], ...]
    """``(operation index, parameter index)`` of every trainable slot."""
    slot_symbol: tuple[int, ...]
    """For each slot, the index of its symbol in ``circuit.parameters``."""
    n_params: int


def _prepare(circuit: Circuit, values: Values) -> _Problem:
    by_name = {key.name if isinstance(key, Parameter) else key: v for key, v in values.items()}
    symbols = circuit.parameters
    missing = [p.name for p in symbols if p.name not in by_name]
    if missing:
        raise ValueError(f"Missing value(s) for parameter(s): {', '.join(missing)}.")
    bound = circuit.bind(by_name)  # raises on unknown names and non-finite values
    index = {p: k for k, p in enumerate(symbols)}
    slots = circuit.parameter_slots
    return _Problem(
        n_wires=circuit.n_wires,
        operations=bound.operations,
        slots=tuple((i, j) for i, j, _ in slots),
        slot_symbol=tuple(index[p] for _, _, p in slots),
        n_params=len(symbols),
    )


def _shifted(ops: tuple[Operation, ...], slot: tuple[int, int], delta: float) -> list[Operation]:
    """Return ``ops`` with the parameter at ``slot`` increased by ``delta``."""
    i, j = slot
    params = list(ops[i].params)
    params[j] = cast(float, params[j]) + delta  # bound circuit: every parameter is a float
    shifted = list(ops)
    shifted[i] = dataclasses.replace(ops[i], params=tuple(params))
    return shifted


def _to_symbols(slot_jacobian: Jacobian, problem: _Problem) -> Jacobian:
    """Sum the columns of slots that share a symbol (chain rule): ``J_slot @ S``.

    ``S[s, p] = 1`` if slot ``s`` holds symbol ``p``.
    """
    selector = np.zeros((len(problem.slots), problem.n_params))
    selector[np.arange(len(problem.slots)), problem.slot_symbol] = 1.0
    result: Jacobian = slot_jacobian @ selector
    return result


def _evaluator(
    problem: _Problem,
    observables: Sequence[Observable],
    measurement: Measurement,
    backend: Backend[Any],
    dtype: DTypeLike,
) -> Callable[[Sequence[Operation]], npt.NDArray[np.float64]]:
    """Return ``ops ↦ [measurement of each observable]``, one circuit execution per call."""
    measure = backend.expval if measurement == "expval" else backend.var

    def evaluate(ops: Sequence[Operation]) -> npt.NDArray[np.float64]:
        state = backend.apply_operations(backend.zero_state(problem.n_wires, dtype), ops)
        return np.array([measure(state, obs) for obs in observables], dtype=np.float64)

    return evaluate


def _finite_diff(
    problem: _Problem,
    evaluate: Callable[[Sequence[Operation]], npt.NDArray[np.float64]],
    n_outputs: int,
    step: float,
) -> Jacobian:
    jac = np.empty((n_outputs, len(problem.slots)))
    for s, slot in enumerate(problem.slots):
        plus = evaluate(_shifted(problem.operations, slot, step))
        minus = evaluate(_shifted(problem.operations, slot, -step))
        jac[:, s] = (plus - minus) / (2 * step)
    return jac


def jacobian(  # noqa: PLR0913  # keyword-only options, as in Circuit.expval
    circuit: Circuit,
    observables: Sequence[Observable],
    values: Values,
    *,
    method: Method = "finite-diff",
    measurement: Measurement = "expval",
    backend: Backend[Any] | None = None,
    dtype: DTypeLike = "complex128",
    step: float | None = None,
) -> Jacobian:
    r"""Return :math:`J_{mp} = \partial \langle O_m \rangle / \partial \theta_p`.

    Parameters
    ----------
    circuit
        Circuit with symbolic parameters.
    observables
        The observables :math:`O_m`.
    values
        A value for every symbol of ``circuit``, keyed by :class:`Parameter` or by name.
    method
        Differentiation method (see the module docstring).
    measurement
        ``"expval"`` (default) or ``"var"``: the quantity differentiated for each observable.
    backend
        Backend to run on (default: :class:`~ketjet.backends.NumPyBackend`).
    dtype
        Precision: ``"complex128"`` (default) or ``"complex64"``.
    step
        Finite-difference step (default: ``1e-7`` for complex128, ``1e-3`` for complex64).

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(len(observables), len(circuit.parameters))``.

    Raises
    ------
    ValueError
        If a symbol has no value, a value names no symbol, the method is unknown, or the method
        doesn't support ``measurement``.

    Examples
    --------
    >>> from ketjet import Circuit
    >>> from ketjet.observables import Z
    >>> from ketjet.operations import Parameter
    >>> a, b = Parameter("a"), Parameter("b")
    >>> c = Circuit(2).rx(a, 0).ry(b, 1)
    >>> jacobian(c, [Z(0), Z(1)], {"a": 0.5, "b": 1.0}, method="finite-diff").round(6)
    array([[-0.479426,  0.      ],
           [ 0.      , -0.841471]])
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(_METHODS)}.")
    if measurement not in _MEASUREMENTS:
        raise ValueError(
            f"Unknown measurement {measurement!r}; choose one of {', '.join(_MEASUREMENTS)}."
        )
    observables = list(observables)
    problem = _prepare(circuit, values)
    engine: Backend[Any] = backend if backend is not None else NumPyBackend()
    evaluate = _evaluator(problem, observables, measurement, engine, dtype)
    if step is None:
        step = _DEFAULT_STEP[np.dtype(as_complex_dtype(dtype))]
    if not step > 0:
        raise ValueError(f"The finite-difference step must be positive, got {step}.")
    slot_jacobian = _finite_diff(problem, evaluate, len(observables), step)
    return _to_symbols(slot_jacobian, problem)


def grad(  # noqa: PLR0913  # keyword-only options, as in Circuit.expval
    circuit: Circuit,
    observable: Observable,
    values: Values,
    *,
    method: Method = "finite-diff",
    measurement: Measurement = "expval",
    backend: Backend[Any] | None = None,
    dtype: DTypeLike = "complex128",
    step: float | None = None,
) -> npt.NDArray[np.float64]:
    r"""Return the gradient :math:`\partial \langle O \rangle / \partial \theta_p`, shape ``(P,)``.

    Equivalent to ``jacobian(circuit, [observable], values, ...)[0]``; see :func:`jacobian`.
    """
    jac = jacobian(
        circuit,
        [observable],
        values,
        method=method,
        measurement=measurement,
        backend=backend,
        dtype=dtype,
        step=step,
    )
    result: npt.NDArray[np.float64] = jac[0]
    return result
