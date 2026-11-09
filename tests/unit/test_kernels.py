"""Tests for the reference apply_matrix kernel (QS-203, ADR-0001)."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet.kernels import apply_matrix, n_wires_of
from ketjet.statevector import zero_state
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]

X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


# --------------------------------------------------------------------------- independent reference


def dense_operator(
    matrix: npt.NDArray[np.complex128], wires: tuple[int, ...], n: int
) -> npt.NDArray[np.complex128]:
    """Build the full 2**n x 2**n operator entry by entry from bit arithmetic.

    Deliberately shares no code or technique with the kernel (no reshapes, no tensordot): entry
    (r, c) is matrix[sub(r), sub(c)] when r and c agree on every non-target wire, else 0, where
    sub() reads the target bits in wire order with wires[0] as the most significant bit.
    """
    size = 2**n
    full = np.zeros((size, size), dtype=np.complex128)

    def bit(index: int, wire: int) -> int:
        return (index >> (n - 1 - wire)) & 1

    def sub(index: int) -> int:
        value = 0
        for w in wires:
            value = (value << 1) | bit(index, w)
        return value

    others = [w for w in range(n) if w not in wires]
    for r in range(size):
        for c in range(size):
            if all(bit(r, w) == bit(c, w) for w in others):
                full[r, c] = matrix[sub(r), sub(c)]
    return full


def random_unitary(dim: int, rng: np.random.Generator) -> npt.NDArray[np.complex128]:
    """Haar-random unitary via QR of a complex Gaussian matrix (phase-corrected)."""
    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phases = np.diag(r) / np.abs(np.diag(r))
    unitary: npt.NDArray[np.complex128] = np.asarray(q * phases, dtype=np.complex128)
    return unitary


def random_state(n: int, rng: np.random.Generator, dtype: Any) -> npt.NDArray[Any]:
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    normalized: npt.NDArray[Any] = (v / np.linalg.norm(v)).astype(dtype)
    return normalized


@st.composite
def gate_case(draw: st.DrawFn, max_n: int = 6, max_k: int = 3) -> tuple[int, tuple[int, ...], int]:
    n = draw(st.integers(min_value=1, max_value=max_n))
    k = draw(st.integers(min_value=1, max_value=min(max_k, n)))
    wires = tuple(draw(st.permutations(range(n)))[:k])
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return n, wires, seed


# --------------------------------------------------------------------------- known results


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_x_on_each_wire_of_three_qubits(wire: int, dtype: Any) -> None:
    out = apply_matrix(zero_state(3, dtype), X, [wire])
    expected = np.zeros(8, dtype=dtype)
    expected[1 << (2 - wire)] = 1  # wire 0 is the most significant bit
    assert_close(out, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_hadamard_on_every_wire_gives_uniform_superposition(dtype: Any) -> None:
    state = zero_state(3, dtype)
    for w in range(3):
        state = apply_matrix(state, H, [w])
    assert_close(state, np.full(8, 1 / np.sqrt(8)), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bell_state(dtype: Any) -> None:
    state = apply_matrix(apply_matrix(zero_state(2, dtype), H, [0]), CNOT, [0, 1])
    assert_close(state, np.array([1, 0, 0, 1]) / np.sqrt(2), dtype)


def test_cnot_wire_order_matters() -> None:
    # |10> (index 2): control on wire 0 flips wire 1 -> |11>; control on wire 1 does nothing.
    basis_10 = np.zeros(4, dtype=complex)
    basis_10[2] = 1
    assert np.argmax(np.abs(apply_matrix(basis_10, CNOT, [0, 1]))) == 3
    assert np.argmax(np.abs(apply_matrix(basis_10, CNOT, [1, 0]))) == 2


def test_non_adjacent_and_reversed_wires_match_reference() -> None:
    rng = np.random.default_rng(7)
    u = random_unitary(4, rng)
    state = random_state(4, rng, np.complex128)
    for wires in [(0, 3), (3, 0), (2, 1)]:
        assert_close(
            apply_matrix(state, u, wires), dense_operator(u, wires, 4) @ state, np.complex128
        )


# --------------------------------------------------------------------------- properties


@settings(max_examples=150, deadline=None)
@given(case=gate_case(), dtype=st.sampled_from(DTYPES))
def test_matches_independent_dense_reference(
    case: tuple[int, tuple[int, ...], int], dtype: Any
) -> None:
    n, wires, seed = case
    rng = np.random.default_rng(seed)
    u = random_unitary(2 ** len(wires), rng)
    state = random_state(n, rng, dtype)
    expected = dense_operator(u, wires, n) @ state.astype(np.complex128)
    assert_close(apply_matrix(state, u, wires), expected, dtype)


@settings(max_examples=100, deadline=None)
@given(case=gate_case(max_n=8), dtype=st.sampled_from(DTYPES))
def test_unitary_preserves_norm(case: tuple[int, tuple[int, ...], int], dtype: Any) -> None:
    n, wires, seed = case
    rng = np.random.default_rng(seed)
    out = apply_matrix(random_state(n, rng, dtype), random_unitary(2 ** len(wires), rng), wires)
    assert_close(np.linalg.norm(out), 1.0, dtype)


@settings(max_examples=100, deadline=None)
@given(case=gate_case(max_n=8), dtype=st.sampled_from(DTYPES))
def test_gate_then_inverse_is_identity(case: tuple[int, tuple[int, ...], int], dtype: Any) -> None:
    n, wires, seed = case
    rng = np.random.default_rng(seed)
    u = random_unitary(2 ** len(wires), rng)
    state = random_state(n, rng, dtype)
    roundtrip = apply_matrix(apply_matrix(state, u, wires), u.conj().T, wires)
    assert_close(roundtrip, state, dtype)


@settings(max_examples=50, deadline=None)
@given(case=gate_case(), dtype=st.sampled_from(DTYPES))
def test_dtype_and_layout_preserved_and_input_untouched(
    case: tuple[int, tuple[int, ...], int], dtype: Any
) -> None:
    n, wires, seed = case
    rng = np.random.default_rng(seed)
    state = random_state(n, rng, dtype)
    before = state.copy()
    out = apply_matrix(state, random_unitary(2 ** len(wires), rng), wires)
    assert out.dtype == np.dtype(dtype)  # complex64 is never promoted (ADR-0001 D3)
    assert out.shape == state.shape
    assert out.flags.c_contiguous
    assert out is not state
    np.testing.assert_array_equal(state, before)  # pure kernel (ADR-0001 D4)


# --------------------------------------------------------------------------- validation


def test_n_wires_of() -> None:
    assert n_wires_of(zero_state(5)) == 5


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (np.zeros((2, 2), dtype=complex), "one-dimensional"),
        (np.zeros(3, dtype=complex), "power of two"),
        (np.zeros(1, dtype=complex), "power of two"),
        (np.zeros(0, dtype=complex), "power of two"),
    ],
)
def test_invalid_states_rejected(state: npt.NDArray[Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_matrix(state, X, [0])


def test_non_complex_state_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        apply_matrix(np.array([1.0, 0.0]), X, [0])


@pytest.mark.parametrize(
    ("wires", "error", "message"),
    [
        ([], ValueError, "At least one wire"),
        ([2], ValueError, "out of range"),
        ([-1], ValueError, "out of range"),
        ([0, 0], ValueError, "distinct"),
        ([0.0], TypeError, "must be ints"),
        ([True], TypeError, "must be ints"),
    ],
)
def test_invalid_wires_rejected(wires: list[Any], error: type[Exception], message: str) -> None:
    matrix = np.eye(2 ** max(len(wires), 1))
    with pytest.raises(error, match=message):
        apply_matrix(zero_state(2), matrix, wires)


@pytest.mark.parametrize("matrix", [np.eye(4), np.eye(2)[:, :1], np.ones(2)])
def test_matrix_shape_must_match_wires(matrix: npt.NDArray[Any]) -> None:
    with pytest.raises(ValueError, match="needs a 2x2 matrix"):
        apply_matrix(zero_state(2), matrix, [0])


def test_numpy_integer_wires_accepted() -> None:
    out = apply_matrix(zero_state(2), X, [np.int64(0)])
    assert out[2] == 1
