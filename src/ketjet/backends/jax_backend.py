r"""The JAX backend: the NumPy oracle's results, on kernels JAX can trace (ADR-0006).

Install it with ``pip install "ketjet[jax]"``. Importing :mod:`ketjet` never imports JAX; this
module does.

The module has two layers (ADR-0006 D6):

- **pure functions of JAX arrays** (:func:`apply_gate`, :func:`probabilities`,
  :func:`apply_observable`, :func:`expectation`, :func:`variance`). Wires, controls and gate names
  are static Python values, and gate parameters may be traced values, so these functions can be
  differentiated and compiled;
- :class:`JaxBackend`, which implements the :class:`~ketjet.backends.Backend` protocol on top of
  them and returns NumPy types wherever the protocol does.

**Precision (ADR-0006 D5).** JAX computes in 32 bits unless its 64-bit mode is on, and would
silently return complex64 when asked for complex128. This backend raises instead, and never
changes the setting itself. Enable it once, before creating states::

    import jax
    jax.config.update("jax_enable_x64", True)

complex64 states work in either mode.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from ketjet.gates import GATES, get_gate
from ketjet.kernels import n_wires_of
from ketjet.measurements import (
    Probabilities,
    Samples,
    Seed,
    _check_shots,
    _measured_wires,
    sample_from_probs,
)
from ketjet.observables import Hermitian, Observable, Pauli, Prod, Sum, _check_in_state
from ketjet.operations import Operation
from ketjet.statevector import (
    ComplexDType,
    DTypeLike,
    StateVector,
    as_complex_dtype,
    validate_n_wires,
)

__all__ = [
    "MATRIX_BUILDERS",
    "JaxBackend",
    "apply_gate",
    "apply_observable",
    "expectation",
    "gate_matrix",
    "probabilities",
    "variance",
]

Params = Sequence[Any]
"""Gate parameters: Python floats or JAX scalars (traced values included)."""

_MatrixBuilder = Callable[[Params], jax.Array]


# --------------------------------------------------------------------------- precision


def _checked_dtype(dtype: DTypeLike) -> ComplexDType:
    """Normalize ``dtype`` and refuse complex128 while JAX's 64-bit mode is off (ADR-0006 D5)."""
    complex_dtype = as_complex_dtype(dtype)
    if complex_dtype == np.complex128 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "A complex128 state needs JAX's 64-bit mode, which is off, so JAX would silently "
            'compute in complex64. Enable it with jax.config.update("jax_enable_x64", True) '
            'before creating states, or ask for dtype="complex64".'
        )
    return complex_dtype


def _n_wires(state: jax.Array) -> int:
    # n_wires_of only reads ``ndim`` and ``shape``, which a JAX array has too.
    return n_wires_of(cast(StateVector, state))


# --------------------------------------------------------------------------- gate matrices
# JAX twins of the parametrized matrices in ketjet.gates (ADR-0006 D3). The test suite checks
# every one against the NumPy registry, and that no parametrized gate is missing.


def _rx(params: Params) -> jax.Array:
    c, s = jnp.cos(params[0] / 2), jnp.sin(params[0] / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]])


def _ry(params: Params) -> jax.Array:
    c, s = jnp.cos(params[0] / 2), jnp.sin(params[0] / 2)
    return jnp.array([[c, -s], [s, c]])


def _rz(params: Params) -> jax.Array:
    p = jnp.exp(-0.5j * params[0])
    return jnp.array([[p, 0], [0, jnp.conj(p)]])


def _phase(params: Params) -> jax.Array:
    return jnp.array([[1, 0], [0, jnp.exp(1j * params[0])]])


def _rot(params: Params) -> jax.Array:
    # Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)
    phi, theta, omega = params
    return _rz((omega,)) @ _ry((theta,)) @ _rz((phi,))


def _u3(params: Params) -> jax.Array:
    theta, phi, delta = params
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array(
        [
            [c, -jnp.exp(1j * delta) * s],
            [jnp.exp(1j * phi) * s, jnp.exp(1j * (phi + delta)) * c],
        ]
    )


MATRIX_BUILDERS: Final[dict[str, _MatrixBuilder]] = {
    "RX": _rx,
    "RY": _ry,
    "RZ": _rz,
    "PhaseShift": _phase,
    "Rot": _rot,
    "U3": _u3,
}
"""``jax.numpy`` matrix builders of the parametrized gates, keyed by gate name."""


def gate_matrix(gate: str, params: Params = (), dtype: DTypeLike = np.complex128) -> jax.Array:
    """Return a base gate's matrix as a JAX array; ``params`` may be traced values.

    Fixed gates use the constant matrices of :data:`ketjet.gates.GATES`.

    Raises
    ------
    KeyError
        If ``gate`` is unknown.
    ValueError
        If the number of parameters is wrong.

    Examples
    --------
    >>> gate_matrix("Z").real.tolist()
    [[1.0, 0.0], [0.0, -1.0]]
    >>> round(float(gate_matrix("RX", (3.141592653589793,), "complex64")[0, 1].imag), 6)
    -1.0
    """
    spec = get_gate(gate)
    if len(params) != spec.num_params:
        raise ValueError(f"{gate} takes {spec.num_params} parameter(s), got {len(params)}.")
    complex_dtype = as_complex_dtype(dtype)
    if spec.num_params:
        return MATRIX_BUILDERS[gate](params).astype(complex_dtype)
    return jnp.asarray(spec.matrix((), complex_dtype))


# --------------------------------------------------------------------------- kernels
# These act on the rank-n tensor view of a state, in which axis w is wire w (ADR-0001).


def _contract(psi: jax.Array, matrix: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """Contract a ``2**k`` square ``matrix`` with ``axes`` of ``psi`` (first axis = MSB)."""
    k = len(axes)
    out = jnp.tensordot(matrix.reshape((2,) * (2 * k)), psi, axes=(tuple(range(k, 2 * k)), axes))
    return jnp.moveaxis(out, tuple(range(k)), axes)


def _scale(psi: jax.Array, diagonal: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """Multiply ``axes`` of ``psi`` by the ``2**k`` entries of ``diagonal`` (first axis = MSB)."""
    k = len(axes)
    ascending = sorted(range(k), key=lambda i: axes[i])
    shape = [1] * psi.ndim
    for axis in axes:
        shape[axis] = 2
    return psi * jnp.transpose(diagonal.reshape((2,) * k), ascending).reshape(shape)


def apply_gate(  # noqa: PLR0913 - an operation's fields, unpacked so parameters can be traced
    state: jax.Array,
    gate: str,
    params: Params,
    wires: Sequence[int],
    *,
    controls: Sequence[int] = (),
    control_values: Sequence[bool] = (),
) -> jax.Array:
    """Apply a base gate to ``wires``, on the subspace where every control matches.

    Parameters
    ----------
    state
        Flat statevector of ``2**n`` amplitudes. JAX arrays are immutable, so it isn't modified.
    gate
        Name of a base gate in :data:`ketjet.gates.GATES`.
    params
        The gate's parameters: Python floats or JAX scalars, traced values included.
    wires
        Target wires, in the order matching the gate's matrix.
    controls
        Control wires. The controlled matrix is never built: the gate acts on the slice of the
        state where the controls match (ADR-0006 D4).
    control_values
        Trigger value of each control. Empty means "all controls must be 1".

    Returns
    -------
    jax.Array
        A new flat statevector with the dtype of ``state``.

    Raises
    ------
    ValueError
        If a wire is outside the state, or the number of parameters is wrong.

    Examples
    --------
    A controlled X, triggered when wire 0 is 0, takes ``|00⟩`` to ``|01⟩``:

    >>> import jax.numpy as jnp
    >>> zero = jnp.zeros(4, dtype="complex64").at[0].set(1)
    >>> apply_gate(zero, "X", (), wires=[1], controls=[0], control_values=[False]).real.tolist()
    [0.0, 1.0, 0.0, 0.0]
    """
    n = _n_wires(state)
    targets, ctrl = tuple(wires), tuple(controls)
    out_of_range = [w for w in ctrl + targets if w >= n]
    if out_of_range:
        raise ValueError(f"{gate} uses wire(s) {out_of_range}, but the state has {n} qubit(s).")
    values = tuple(control_values) or (True,) * len(ctrl)
    matrix = gate_matrix(gate, params, state.dtype)
    kernel = _scale if GATES[gate].is_diagonal else _contract
    operand = jnp.diagonal(matrix) if GATES[gate].is_diagonal else matrix

    psi = state.reshape((2,) * n)
    if not ctrl:
        return kernel(psi, operand, targets).reshape(-1)
    index: list[int | slice] = [slice(None)] * n
    for wire, value in zip(ctrl, values, strict=True):
        index[wire] = int(value)
    active = tuple(index)
    free = [w for w in range(n) if w not in ctrl]  # the axes left after fixing the controls
    block = kernel(psi[active], operand, tuple(free.index(w) for w in targets))
    return psi.at[active].set(block).reshape(-1)


# --------------------------------------------------------------------------- measurements


def probabilities(state: jax.Array, wires: Sequence[int] | None = None) -> jax.Array:
    """Return the measurement probabilities of ``wires`` as a JAX array (differentiable).

    Same conventions as :func:`ketjet.measurements.probs`: ``wires[0]`` is the most significant
    bit of the result index, and the other wires are summed over.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> plus_zero = jnp.array([1, 0, 1, 0], dtype="complex64") / jnp.sqrt(2)
    >>> probabilities(plus_zero, wires=[1]).round(3).tolist()
    [1.0, 0.0]
    """
    n = _n_wires(state)
    measured = _measured_wires(wires, n)
    p = (state.real**2 + state.imag**2).reshape((2,) * n)
    others = tuple(w for w in range(n) if w not in measured)
    marginal = p.sum(axis=others) if others else p
    # After summing, the remaining axes are the measured wires in ascending order; reorder them.
    order = [sorted(measured).index(w) for w in measured]
    return jnp.transpose(marginal, order).reshape(-1)


_Y_PHASES: Final = (-1j, 1j)  # Y|1⟩ = -i|0⟩ and Y|0⟩ = i|1⟩: the phases of the flipped state


def _along(values: Sequence[complex], axis: int, like: jax.Array) -> jax.Array:
    shape = [1] * like.ndim
    shape[axis] = 2
    return jnp.asarray(values, dtype=like.dtype).reshape(shape)


def _apply_observable(psi: jax.Array, observable: Observable) -> jax.Array:
    if isinstance(observable, Pauli):
        for wire, letter in observable.word:
            if letter == "Z":
                psi = psi * _along((1, -1), wire, psi)
            else:
                psi = jnp.flip(psi, axis=wire)
                if letter == "Y":
                    psi = psi * _along(_Y_PHASES, wire, psi)
        return psi
    if isinstance(observable, Hermitian):
        matrix = jnp.asarray(observable.base_matrix, dtype=psi.dtype)
        return _contract(psi, matrix, observable.target_wires)
    if isinstance(observable, Prod):
        # Factors act on disjoint wires, so they commute: apply them in sequence.
        for factor in observable.factors:
            psi = _apply_observable(psi, factor)
        return psi
    if isinstance(observable, Sum):
        total = jnp.zeros_like(psi)
        for coeff, term in observable.terms:
            total = total + coeff * _apply_observable(psi, term)
        return total
    raise TypeError(f"The JAX backend can't apply a {type(observable).__name__} observable.")


def apply_observable(state: jax.Array, observable: Observable) -> jax.Array:
    r"""Return :math:`O|\psi\rangle` as a new flat JAX array (differentiable).

    Pauli words flip and sign the wires they act on, without building a matrix.

    Raises
    ------
    ValueError
        If the observable acts on a wire outside the state.
    TypeError
        If the observable isn't a ``Pauli``, ``Hermitian``, ``Prod`` or ``Sum``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ketjet.observables import X
    >>> apply_observable(jnp.array([1, 0], dtype="complex64"), X(0)).real.tolist()
    [0.0, 1.0]
    """
    n = _n_wires(state)
    _check_in_state(observable, n)
    return _apply_observable(state.reshape((2,) * n), observable).reshape(-1)


def expectation(state: jax.Array, observable: Observable) -> jax.Array:
    r"""Return :math:`\langle\psi|O|\psi\rangle` as a real JAX scalar (differentiable).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ketjet.observables import Z
    >>> float(expectation(jnp.array([0, 1], dtype="complex64"), Z(0)))
    -1.0
    """
    return jnp.vdot(state, apply_observable(state, observable)).real


def variance(state: jax.Array, observable: Observable) -> jax.Array:
    r"""Return :math:`\lVert O\psi \rVert^2 - \langle O \rangle^2`, clipped at 0, as a JAX scalar.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from ketjet.observables import X
    >>> float(variance(jnp.array([1, 0], dtype="complex64"), X(0)))
    1.0
    """
    applied = apply_observable(state, observable)
    mean = jnp.vdot(state, applied).real
    return jnp.maximum(jnp.vdot(applied, applied).real - mean * mean, 0.0)


# --------------------------------------------------------------------------- backend


class JaxBackend:
    """JAX statevector backend: states are immutable ``jax.Array`` objects.

    It returns the NumPy backend's results at the standard tolerances, in both precisions. Every
    method that returns a measurement returns NumPy types, as the
    :class:`~ketjet.backends.Backend` protocol says; the module-level functions return JAX
    arrays and can be differentiated.

    Examples
    --------
    >>> from ketjet.operations import CNOT, H
    >>> backend = JaxBackend()
    >>> bell = backend.apply_operations(backend.zero_state(2, "complex64"), [H(0), CNOT(0, 1)])
    >>> backend.probs(bell).round(3).tolist()
    [0.5, 0.0, 0.0, 0.5]
    """

    name = "jax"

    def zero_state(self, n_wires: int, dtype: DTypeLike = np.complex128) -> jax.Array:
        """Allocate the ``|0…0⟩`` state of ``n_wires`` qubits.

        Raises
        ------
        RuntimeError
            If ``dtype`` is complex128 while JAX's 64-bit mode is off (see the module docstring).
        """
        n = validate_n_wires(n_wires)
        return jnp.zeros(1 << n, dtype=_checked_dtype(dtype)).at[0].set(1)

    def n_wires(self, state: jax.Array) -> int:
        """Return the number of qubits of ``state``."""
        return _n_wires(state)

    def apply_operation(self, state: jax.Array, op: Operation) -> jax.Array:
        """Apply one operation, controls included, and return the new state.

        Raises
        ------
        ValueError
            If the operation uses a wire outside the state, or has unbound symbolic parameters.
        """
        return apply_gate(
            state,
            op.gate,
            op._numeric_params(),
            op.wires,
            controls=op.controls,
            control_values=op.control_values,
        )

    def apply_operations(self, state: jax.Array, ops: Iterable[Operation]) -> jax.Array:
        """Apply operations in order and return the final state."""
        for op in ops:
            state = self.apply_operation(state, op)
        return state

    def to_numpy(self, state: jax.Array) -> StateVector:
        """Return a copy of the amplitudes on the host (the caller may modify it freely)."""
        return np.array(state)

    def probs(self, state: jax.Array, wires: Sequence[int] | None = None) -> Probabilities:
        """Return measurement probabilities of ``wires`` (see :func:`probabilities`)."""
        return np.array(probabilities(state, wires))

    def expval(self, state: jax.Array, observable: Observable) -> float:
        """Return the expectation value of ``observable``."""
        return float(expectation(state, observable))

    def var(self, state: jax.Array, observable: Observable) -> float:
        """Return the variance of ``observable``."""
        return float(variance(state, observable))

    def sample(
        self, state: jax.Array, shots: int, wires: Sequence[int] | None = None, seed: Seed = None
    ) -> Samples:
        """Return measured bits, drawn on the host by ketjet's seeded sampler (ADR-0006 D6).

        Only the probabilities of the measured wires leave JAX. Given the same probabilities,
        a seed gives the same bits as the NumPy backend.
        """
        n_shots = _check_shots(shots)
        measured = _measured_wires(wires, _n_wires(state))
        p = np.asarray(probabilities(state, measured))
        return sample_from_probs(p, n_shots, len(measured), seed)

    def adjoint_jacobian(
        self,
        n_wires: int,
        operations: Sequence[Operation],
        observables: Sequence[Observable],
        trainable: Sequence[tuple[int, int]],
        *,
        dtype: DTypeLike = np.complex128,
    ) -> npt.NDArray[np.float64]:
        """Return ``∂⟨O_m⟩/∂x_s`` for the ``trainable`` slots, by reverse-mode autodiff.

        Same arguments, shape and values as :func:`ketjet.adjoint.adjoint_jacobian`.

        Notes
        -----
        This uses ``jax.jacrev``, which stores the forward pass, so its memory grows with
        circuit depth: **the depth-independent memory of ADR-0004 D4 isn't met on this backend
        yet**. QS-606 replaces it with the adjoint recursion (ADR-0006 D7). Use the NumPy
        backend where that bound matters.

        Raises
        ------
        ValueError
            If a trainable slot names an operation or parameter that doesn't exist.
        RuntimeError
            If ``dtype`` is complex128 while JAX's 64-bit mode is off.
        """
        complex_dtype = _checked_dtype(dtype)
        ops = list(operations)
        for i, j in trainable:
            if not 0 <= i < len(ops) or not 0 <= j < ops[i].spec.num_params:
                raise ValueError(f"Trainable slot {(i, j)} doesn't exist in the circuit.")
        jac = np.zeros((len(observables), len(trainable)))
        if not trainable or not observables:
            return jac

        values = [op._numeric_params() for op in ops]
        # A slot named twice is differentiated once, and its column is repeated.
        column = {slot: k for k, slot in enumerate(dict.fromkeys(trainable))}
        real_dtype = np.float64 if complex_dtype == np.complex128 else np.float32
        theta = jnp.asarray([values[i][j] for i, j in column], dtype=real_dtype)

        def measure(x: jax.Array) -> jax.Array:
            state = self.zero_state(n_wires, complex_dtype)
            for i, op in enumerate(ops):
                params = [
                    x[column[i, j]] if (i, j) in column else value
                    for j, value in enumerate(values[i])
                ]
                state = apply_gate(
                    state,
                    op.gate,
                    params,
                    op.wires,
                    controls=op.controls,
                    control_values=op.control_values,
                )
            return jnp.stack([expectation(state, obs) for obs in observables])

        unique = np.asarray(jax.jacrev(measure)(theta), dtype=np.float64)
        jac[:] = unique[:, [column[slot] for slot in trainable]]
        return jac
