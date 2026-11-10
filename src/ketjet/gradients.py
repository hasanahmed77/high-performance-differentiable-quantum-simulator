r"""Gradients of circuit expectation values (ADR-0004).

Gradients are taken with respect to a circuit's :class:`~ketjet.operations.Parameter` symbols, in
:attr:`Circuit.parameters <ketjet.circuit.Circuit.parameters>` order. Every method first
differentiates each *slot* (one gate parameter holding a symbol), then sums the slots that share a
symbol, which is the chain rule (ADR-0004 D1).

Methods
-------
``"adjoint"`` (default)
    Exact, for expectation values. One forward and one backward pass, whatever the number of
    parameters, with memory independent of circuit depth (ADR-0004 D3). Runs on the backend.
``"parameter-shift"``
    Exact. Each slot is shifted by the rule derived from its gate's generator spectrum
    (ADR-0004 D5): two evaluations for most gates, four for controlled rotations. Works with
    ``shots``, as on hardware.
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
>>> g = grad(Circuit(1).ry(theta, 0), Z(0), {"theta": 0.3})
>>> bool(np.isclose(g[0], -np.sin(0.3), rtol=0, atol=1e-12))
True
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, TypedDict, Unpack, cast

import numpy as np
import numpy.typing as npt

from ketjet import measurements
from ketjet.backends.base import Backend
from ketjet.backends.numpy_backend import NumPyBackend
from ketjet.circuit import Circuit
from ketjet.measurements import Seed
from ketjet.observables import Observable
from ketjet.operations import Operation, Parameter
from ketjet.statevector import DTypeLike, as_complex_dtype

__all__ = ["GradientOptions", "Measurement", "Method", "Values", "grad", "jacobian", "shift_rule"]

Method = Literal["adjoint", "parameter-shift", "finite-diff"]
"""A differentiation method name."""

Measurement = Literal["expval", "var"]
"""The differentiated quantity of each observable."""

Values = Mapping[Parameter, float] | Mapping[str, float]
"""Values for every symbol of a circuit, keyed by :class:`Parameter` or by name."""

Jacobian = npt.NDArray[np.float64]

_METHODS: Final[tuple[str, ...]] = ("adjoint", "parameter-shift", "finite-diff")
_MEASUREMENTS: Final[tuple[str, ...]] = ("expval", "var")
_SUPPORTED: Final[dict[str, frozenset[str]]] = {
    "adjoint": frozenset({"expval"}),
    "parameter-shift": frozenset({"expval"}),
    "finite-diff": frozenset({"expval", "var"}),
}
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


_Evaluate = Callable[[Sequence[Operation]], npt.NDArray[np.float64]]


def _evaluator(  # noqa: PLR0913  # private helper; every argument is needed
    n_wires: int,
    observables: Sequence[Observable],
    *,
    measurement: Measurement,
    backend: Backend[Any],
    dtype: DTypeLike,
    shots: int | None,
    seed: Seed,
) -> _Evaluate:
    """Return ``ops ↦ [measurement of each observable]``, one circuit execution per call.

    With ``shots``, every estimate draws from its own child of ``seed`` (ADR-0004 D5), so the
    whole computation is reproducible and the estimates are independent.
    """
    measure = backend.expval if measurement == "expval" else backend.var
    rng = np.random.default_rng(seed) if shots is not None else None

    def evaluate(ops: Sequence[Operation]) -> npt.NDArray[np.float64]:
        state = backend.apply_operations(backend.zero_state(n_wires, dtype), ops)
        if shots is None or rng is None:  # (rng is None exactly when shots is None)
            return np.array([measure(state, obs) for obs in observables], dtype=np.float64)
        # Shot estimates run on the host copy until QS-903 (see Circuit.expval).
        host = backend.to_numpy(state)
        children = rng.spawn(len(observables))
        return np.array(
            [
                measurements.estimate_expval(host, obs, shots, child)
                for obs, child in zip(observables, children, strict=True)
            ],
            dtype=np.float64,
        )

    return evaluate


@functools.cache
def shift_rule(freqs: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    r"""Return ``(coefficient, shift)`` pairs of the exact parameter-shift rule (ADR-0004 D5).

    For frequencies :math:`\Omega, 2\Omega, \ldots, R\Omega` (Wierichs et al., 2022),

    .. math::

        f'(x) = \sum_{\mu=1}^{R} c_\mu \big[f(x + s_\mu) - f(x - s_\mu)\big],
        \quad s_\mu = \frac{(2\mu - 1)\pi}{2R\Omega},
        \quad c_\mu = \frac{\Omega\,(-1)^{\mu-1}}{4R\sin^2\!\big(\tfrac{(2\mu-1)\pi}{4R}\big)}.

    Parameters
    ----------
    freqs
        Equidistant frequencies, as returned by :func:`ketjet.gates.frequencies`. Empty means the
        output doesn't depend on the parameter.

    Examples
    --------
    A rotation has the two-term rule; a controlled rotation needs four terms:

    >>> [(round(c, 6), round(s, 6)) for c, s in shift_rule((1.0,))]
    [(0.5, 1.570796)]
    >>> [(round(c, 6), round(s, 6)) for c, s in shift_rule((0.5, 1.0))]
    [(0.426777, 1.570796), (-0.073223, 4.712389)]
    """
    r = len(freqs)
    if r == 0:
        return ()
    omega = freqs[0]
    rule = []
    for mu in range(1, r + 1):
        x = (2 * mu - 1) * np.pi / (2 * r)
        coefficient = omega * (-1) ** (mu - 1) / (4 * r * np.sin(x / 2) ** 2)
        rule.append((float(coefficient), float(x / omega)))
    return tuple(rule)


def _slot_rule(op: Operation, param_index: int) -> tuple[tuple[float, float], ...]:
    return shift_rule(op.spec.frequencies(param_index, controlled=bool(op.controls)))


def _parameter_shift(problem: _Problem, evaluate: _Evaluate, n_outputs: int) -> Jacobian:
    jac = np.zeros((n_outputs, len(problem.slots)))
    for s, (i, j) in enumerate(problem.slots):
        for coefficient, shift in _slot_rule(problem.operations[i], j):
            plus = evaluate(_shifted(problem.operations, (i, j), shift))
            minus = evaluate(_shifted(problem.operations, (i, j), -shift))
            jac[:, s] += coefficient * (plus - minus)
    return jac


def _finite_diff(problem: _Problem, evaluate: _Evaluate, n_outputs: int, step: float) -> Jacobian:
    jac = np.empty((n_outputs, len(problem.slots)))
    for s, slot in enumerate(problem.slots):
        plus = evaluate(_shifted(problem.operations, slot, step))
        minus = evaluate(_shifted(problem.operations, slot, -step))
        jac[:, s] = (plus - minus) / (2 * step)
    return jac


class GradientOptions(TypedDict, total=False):
    """Keyword options shared by :func:`jacobian` and :func:`grad`."""

    method: Method
    measurement: Measurement
    backend: Backend[Any] | None
    dtype: DTypeLike
    shots: int | None
    seed: Seed
    step: float | None


def _check_options(method: str, measurement: str, shots: int | None) -> None:
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(_METHODS)}.")
    if measurement not in _MEASUREMENTS:
        raise ValueError(
            f"Unknown measurement {measurement!r}; choose one of {', '.join(_MEASUREMENTS)}."
        )
    if measurement not in _SUPPORTED[method]:
        able = next(m for m in _METHODS if measurement in _SUPPORTED[m])
        raise ValueError(
            f"method={method!r} can't differentiate {measurement!r}; use method={able!r}."
        )
    if shots is not None and method != "parameter-shift":
        raise ValueError(
            f"method={method!r} can't use shots; use method='parameter-shift' for shot-based "
            "gradients."
        )


def jacobian(  # noqa: PLR0913  # keyword-only options, as in Circuit.expval
    circuit: Circuit,
    observables: Sequence[Observable],
    values: Values,
    *,
    method: Method = "adjoint",
    measurement: Measurement = "expval",
    backend: Backend[Any] | None = None,
    dtype: DTypeLike = "complex128",
    shots: int | None = None,
    seed: Seed = None,
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
        Only ``"finite-diff"`` supports ``"var"``.
    backend
        Backend to run on (default: :class:`~ketjet.backends.NumPyBackend`).
    dtype
        Precision: ``"complex128"`` (default) or ``"complex64"``.
    shots, seed
        Estimate every shifted expectation value from ``shots`` measurements, reproducibly for a
        given ``seed`` (``"parameter-shift"`` only). ``None`` (default) is exact.
    step
        Finite-difference step (default: ``1e-7`` for complex128, ``1e-3`` for complex64).

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(len(observables), len(circuit.parameters))``.

    Raises
    ------
    ValueError
        If a symbol has no value, a value names no symbol, the method or measurement is unknown,
        or the method doesn't support the measurement or ``shots``.

    Examples
    --------
    >>> from ketjet import Circuit
    >>> from ketjet.observables import Z
    >>> from ketjet.operations import Parameter
    >>> a, b = Parameter("a"), Parameter("b")
    >>> c = Circuit(2).rx(a, 0).ry(b, 1)
    >>> jacobian(c, [Z(0), Z(1)], {"a": 0.5, "b": 1.0}).round(6)
    array([[-0.479426,  0.      ],
           [ 0.      , -0.841471]])
    """
    _check_options(method, measurement, shots)
    observables = list(observables)
    problem = _prepare(circuit, values)
    engine: Backend[Any] = backend if backend is not None else NumPyBackend()
    if method == "adjoint":
        slot_jacobian = engine.adjoint_jacobian(
            problem.n_wires, problem.operations, observables, problem.slots, dtype=dtype
        )
        return _to_symbols(np.asarray(slot_jacobian, dtype=np.float64), problem)
    evaluate = _evaluator(
        problem.n_wires,
        observables,
        measurement=measurement,
        backend=engine,
        dtype=dtype,
        shots=shots,
        seed=seed,
    )
    if method == "parameter-shift":
        slot_jacobian = _parameter_shift(problem, evaluate, len(observables))
    else:
        if step is None:
            step = _DEFAULT_STEP[np.dtype(as_complex_dtype(dtype))]
        if not step > 0:
            raise ValueError(f"The finite-difference step must be positive, got {step}.")
        slot_jacobian = _finite_diff(problem, evaluate, len(observables), step)
    return _to_symbols(slot_jacobian, problem)


def grad(
    circuit: Circuit,
    observable: Observable,
    values: Values,
    **options: Unpack[GradientOptions],
) -> npt.NDArray[np.float64]:
    r"""Return the gradient :math:`\partial \langle O \rangle / \partial \theta_p`, shape ``(P,)``.

    Equivalent to ``jacobian(circuit, [observable], values, **options)[0]``; see
    :func:`jacobian` for the options.
    """
    result: npt.NDArray[np.float64] = jacobian(circuit, [observable], values, **options)[0]
    return result
