"""Specialized diagonal and permutation kernels equal the dense kernel (QS-205)."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet import kernels
from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from ketjet.gates import GATES
from ketjet.kernels import apply_diagonal, apply_matrix, apply_permutation, apply_structured
from ketjet.operations import Operation
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
DIAGONAL = sorted(name for name, spec in GATES.items() if spec.is_diagonal)
PERMUTATION = sorted(name for name, spec in GATES.items() if spec.is_permutation)
STRUCTURED = sorted(set(DIAGONAL) | set(PERMUTATION))
MAX_WIRES = 6


def _random_state(n: int, rng: np.random.Generator, dtype: Any) -> npt.NDArray[Any]:
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    normalized: npt.NDArray[Any] = (psi / np.linalg.norm(psi)).astype(dtype)
    return normalized


def _random_operation(name: str, rng: np.random.Generator) -> tuple[Operation, int]:
    """``name`` on random wires, with up to 2 random controls; also returns a fitting wire count."""
    spec = GATES[name]
    n_controls = int(rng.integers(0, 3))
    n = int(rng.integers(spec.num_wires + n_controls, MAX_WIRES + 1))
    chosen = [int(w) for w in rng.permutation(n)[: spec.num_wires + n_controls]]
    params = tuple(float(p) for p in rng.uniform(-np.pi, np.pi, spec.num_params))
    values = tuple(bool(v) for v in rng.integers(0, 2, n_controls))
    op = Operation(name, tuple(chosen[n_controls:]), params, tuple(chosen[:n_controls]), values)
    return op, n


def _random_phased_permutation(
    dim: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    perm = rng.permutation(dim)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, dim))
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    matrix[perm, np.arange(dim)] = phases
    return perm, phases, matrix


# --------------------------------------------------------------------------- every structured gate


def test_structured_gates_exist() -> None:
    # Guards the parametrization below against an empty list.
    assert {"Z", "RZ", "PhaseShift", "T"} <= set(DIAGONAL)
    assert {"X", "SWAP", "ISWAP"} <= set(PERMUTATION)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", STRUCTURED)
@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_structured_kernel_equals_dense_for_every_gate(name: str, dtype: Any, seed: int) -> None:
    """Random parameters, wires, controls and control values, in both precisions."""
    rng = np.random.default_rng(seed)
    op, n = _random_operation(name, rng)
    state = _random_state(n, rng, dtype)
    before = state.copy()
    matrix = op.matrix(dtype)
    expected = apply_matrix(state, matrix, op.all_wires)
    for flags in (
        {"diagonal": op.is_diagonal, "permutation": op.is_permutation},  # what backends use
        {"diagonal": False, "permutation": op.is_permutation},  # the permutation path alone
    ):
        result = apply_structured(state, matrix, op.all_wires, **flags)
        assert result.dtype == state.dtype
        assert result.flags.c_contiguous
        assert_close(result, expected, dtype)
    np.testing.assert_array_equal(state, before)  # pure


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", DIAGONAL)
def test_diagonal_gate_derivative_uses_the_diagonal_kernel(name: str, dtype: Any) -> None:
    """The adjoint pass applies diagonal gates' derivatives with apply_diagonal."""
    rng = np.random.default_rng(len(name))
    for _ in range(10):
        op, n = _random_operation(name, rng)
        state = _random_state(n, rng, dtype)
        for j in range(op.spec.num_params):
            d = op.derivative_matrix(j, dtype)
            assert_close(
                apply_structured(state, d, op.all_wires, diagonal=True),
                apply_matrix(state, d, op.all_wires),
                dtype,
            )


@pytest.mark.parametrize(
    ("op", "kernel"),
    [
        (ops.CZ(0, 2), "apply_diagonal"),
        (ops.RZ(0.3, 1), "apply_diagonal"),
        (ops.T(0), "apply_diagonal"),  # diagonal takes precedence over permutation
        (ops.CNOT(2, 0), "apply_permutation"),
        (ops.Toffoli(0, 1, 2), "apply_permutation"),
        (ops.SWAP(0, 2), "apply_permutation"),
        (ops.H(1), "apply_matrix"),
        (ops.CRX(0.4, 0, 1), "apply_matrix"),
    ],
)
def test_backend_dispatches_to_the_specialized_kernel(
    op: Operation, kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    for name in ("apply_diagonal", "apply_permutation", "apply_matrix"):
        original = getattr(kernels, name)

        def spy(*args: Any, _name: str = name, _original: Any = original, **kw: Any) -> Any:
            calls.append(_name)
            return _original(*args, **kw)

        monkeypatch.setattr(kernels, name, spy)
    NumPyBackend().apply_operation(_random_state(3, np.random.default_rng(0), complex), op)
    assert calls == [kernel]


# --------------------------------------------------------------------------- arbitrary structure


@pytest.mark.parametrize("dtype", DTYPES)
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), k=st.integers(1, 3), data=st.data())
def test_random_diagonal_equals_dense(seed: int, k: int, dtype: Any, data: st.DataObject) -> None:
    rng = np.random.default_rng(seed)
    n = data.draw(st.integers(k, MAX_WIRES))
    wires = [int(w) for w in rng.permutation(n)[:k]]
    # Not unitary, with exact ones mixed in, so the "skip entries equal to 1" path is covered.
    diagonal = rng.normal(size=2**k) + 1j * rng.normal(size=2**k)
    diagonal[rng.random(2**k) < 0.4] = 1.0
    state = _random_state(n, rng, dtype)
    assert_close(
        apply_diagonal(state, diagonal, wires),
        apply_matrix(state, np.diag(diagonal), wires),
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), k=st.integers(1, 3), data=st.data())
def test_random_phased_permutation_equals_dense(
    seed: int, k: int, dtype: Any, data: st.DataObject
) -> None:
    rng = np.random.default_rng(seed)
    n = data.draw(st.integers(k, MAX_WIRES))
    wires = [int(w) for w in rng.permutation(n)[:k]]
    perm, phases, matrix = _random_phased_permutation(2**k, rng)
    if data.draw(st.booleans()):
        phases[rng.random(2**k) < 0.5] = 1.0
        matrix[perm, np.arange(2**k)] = phases
    state = _random_state(n, rng, dtype)
    expected = apply_matrix(state, matrix, wires)
    assert_close(apply_permutation(state, perm, phases, wires), expected, dtype)
    assert_close(apply_structured(state, matrix, wires, permutation=True), expected, dtype)


def test_identity_returns_an_equal_copy() -> None:
    state = _random_state(3, np.random.default_rng(0), np.complex128)
    for result in (
        apply_diagonal(state, np.ones(4), [0, 2]),
        apply_permutation(state, np.arange(4), np.ones(4), [2, 1]),
    ):
        np.testing.assert_array_equal(result, state)
        assert result is not state
        assert not np.shares_memory(result, state)


def test_kernels_keep_complex64() -> None:
    state = _random_state(2, np.random.default_rng(1), np.complex64)
    assert apply_diagonal(state, [1, 1j], [0]).dtype == np.complex64
    assert apply_permutation(state, [1, 0], [1, 1j], [1]).dtype == np.complex64


def test_non_contiguous_input() -> None:
    rng = np.random.default_rng(2)
    wide = _random_state(4, rng, np.complex128)
    state = wide[::2]  # a strided 3-qubit view
    perm, phases, matrix = _random_phased_permutation(4, rng)
    assert_close(
        apply_permutation(state, perm, phases, [2, 0]),
        apply_matrix(state, matrix, [2, 0]),
        np.complex128,
    )
    assert_close(
        apply_diagonal(state, phases, [1, 2]),
        apply_matrix(state, np.diag(phases), [1, 2]),
        np.complex128,
    )


# --------------------------------------------------------------------------- errors


def test_apply_diagonal_rejects_bad_input() -> None:
    state = np.zeros(4, dtype=complex)
    with pytest.raises(ValueError, match=r"needs 4 diagonal entries, got shape \(2,\)"):
        apply_diagonal(state, [1, 1], [0, 1])
    with pytest.raises(ValueError, match="out of range"):
        apply_diagonal(state, [1, 1], [2])
    with pytest.raises(ValueError, match="distinct"):
        apply_diagonal(state, [1, 1, 1, 1], [0, 0])


@pytest.mark.parametrize(
    ("permutation", "phases", "match"),
    [
        ([0, 0], [1, 1], "must be a permutation of 0 … 1"),
        ([0, 2], [1, 1], "must be a permutation of 0 … 1"),
        ([0, 1, 2], [1, 1], "must be a permutation of 0 … 1"),
        ([0.0, 1.0], [1, 1], "must be a permutation of 0 … 1"),
        ([1, 0], [1, 1, 1], r"needs 2 phases, got \(3,\)"),
    ],
)
def test_apply_permutation_rejects_bad_input(permutation: Any, phases: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_permutation(np.zeros(4, dtype=complex), permutation, phases, [1])


def test_apply_structured_rejects_non_square_matrices() -> None:
    state = np.zeros(4, dtype=complex)
    for flags in ({"diagonal": True}, {"permutation": True}):
        with pytest.raises(ValueError, match=r"square matrix, got shape \(2, 3\)"):
            apply_structured(state, np.ones((2, 3)), [0], **flags)
        with pytest.raises(ValueError, match=r"square matrix, got shape \(2,\)"):
            apply_structured(state, np.ones(2), [0], **flags)


def test_apply_structured_without_flags_is_the_dense_kernel() -> None:
    rng = np.random.default_rng(3)
    state = _random_state(2, rng, np.complex128)
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    np.testing.assert_array_equal(apply_structured(state, m, [1]), apply_matrix(state, m, [1]))
