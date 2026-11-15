"""Shared helpers for differential tests against PennyLane.

PennyLane objects are built from PennyLane's own native gates and observables (never from our
matrices), so comparisons are independent. Import this only after
``pytest.importorskip("pennylane")``.
"""

import importlib.util
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pennylane as qml

from ketjet.observables import Hermitian, Observable, Pauli, Prod, Sum
from ketjet.operations import Operation
from tests.random_circuits import random_circuit

__all__ = [
    "PENNYLANE_JAX_DEPRECATIONS",
    "PENNYLANE_JAX_ONLY",
    "PENNYLANE_JAX_WORKS",
    "random_circuit",
    "run_default_qubit",
    "to_pennylane",
    "to_pennylane_observable",
]


def _pennylane_jax_works() -> bool:
    """PennyLane 0.45 calls ``jax.core.is_concrete``, which JAX removed in 0.11 (ADR-0006 D1)."""
    if not importlib.util.find_spec("jax"):
        return False
    import jax  # noqa: PLC0415 - JAX is optional

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # deprecated on JAX 0.10
        return hasattr(jax.core, "is_concrete")


PENNYLANE_JAX_WORKS = _pennylane_jax_works()
"""``True`` if PennyLane's JAX code paths work with the installed JAX."""

PENNYLANE_JAX_ONLY = (
    "PennyLane 0.45's JAX code needs JAX < 0.11 (ADR-0006 D1); run in CI's conformance job"
)
"""Skip reason for tests that need :data:`PENNYLANE_JAX_WORKS`."""

PENNYLANE_JAX_DEPRECATIONS = r"ignore:jax\.core\.\w+ is deprecated:DeprecationWarning"
"""Warning filter for the ``jax.core`` functions PennyLane 0.45 calls, deprecated in JAX 0.10."""


_PL_BASE: dict[str, Callable[..., Any]] = {
    "I": lambda w, p: qml.Identity(w[0]),
    "X": lambda w, p: qml.PauliX(w[0]),
    "Y": lambda w, p: qml.PauliY(w[0]),
    "Z": lambda w, p: qml.PauliZ(w[0]),
    "H": lambda w, p: qml.Hadamard(w[0]),
    "S": lambda w, p: qml.S(w[0]),
    "Sdg": lambda w, p: qml.adjoint(qml.S(w[0])),
    "T": lambda w, p: qml.T(w[0]),
    "Tdg": lambda w, p: qml.adjoint(qml.T(w[0])),
    "SX": lambda w, p: qml.SX(w[0]),
    "SXdg": lambda w, p: qml.adjoint(qml.SX(w[0])),
    "RX": lambda w, p: qml.RX(p[0], w[0]),
    "RY": lambda w, p: qml.RY(p[0], w[0]),
    "RZ": lambda w, p: qml.RZ(p[0], w[0]),
    "PhaseShift": lambda w, p: qml.PhaseShift(p[0], w[0]),
    "Rot": lambda w, p: qml.Rot(*p, w[0]),
    "U3": lambda w, p: qml.U3(*p, w[0]),
    "SWAP": lambda w, p: qml.SWAP(list(w)),
    "ISWAP": lambda w, p: qml.ISWAP(list(w)),
    "ISWAPdg": lambda w, p: qml.adjoint(qml.ISWAP(list(w))),
}
_PL_PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def to_pennylane(op: Operation, params: Sequence[Any] | None = None) -> Any:
    """Build the equivalent PennyLane operation (queued if called inside a QNode).

    ``params`` overrides ``op.params``, e.g. with trainable PennyLane values.
    """
    base = _PL_BASE[op.gate](op.wires, op.params if params is None else tuple(params))
    if not op.controls:
        return base
    return qml.ctrl(base, control=list(op.controls), control_values=list(op.control_values))


def to_pennylane_observable(obs: Observable) -> Any:
    """Build the equivalent PennyLane observable."""
    if isinstance(obs, Pauli):
        if not obs.word:
            return qml.Identity(0)
        factors = [_PL_PAULI[letter](w) for w, letter in obs.word]
        return factors[0] if len(factors) == 1 else qml.prod(*factors)
    if isinstance(obs, Hermitian):
        return qml.Hermitian(np.asarray(obs.base_matrix), wires=list(obs.target_wires))
    if isinstance(obs, Prod):
        return qml.prod(*(to_pennylane_observable(f) for f in obs.factors))
    if isinstance(obs, Sum):
        return qml.sum(*(c * to_pennylane_observable(t) for c, t in obs.terms))
    raise TypeError(type(obs).__name__)  # pragma: no cover


def run_default_qubit(
    circuit: Sequence[Operation], n_wires: int, measurement: Callable[[], Any]
) -> Any:
    """Run ``circuit`` on ``default.qubit`` and return ``measurement()`` (e.g. ``qml.state``)."""
    device = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(device)  # type: ignore[untyped-decorator]  # PennyLane ships no type hints
    def run() -> Any:
        for op in circuit:
            # Constructing a PennyLane op inside a QNode queues it; don't also call qml.apply.
            to_pennylane(op)
        return measurement()

    return run()
