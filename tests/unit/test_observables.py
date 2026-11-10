"""Tests for observables: algebra, validation and the matrix-free Pauli path (QS-305, ADR-0003)."""

from functools import reduce
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ketjet.observables import Hermitian, I, Observable, Pauli, Prod, Sum, X, Y, Z
from tests.tolerances import assert_close

DTYPES = [np.complex128, np.complex64]
_P = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def dense_pauli(word: dict[int, str], n: int) -> npt.NDArray[np.complex128]:
    """Independent reference: Kronecker product in wire order (wire 0 = most significant)."""
    return reduce(np.kron, [_P[word.get(w, "I")] for w in range(n)]).astype(np.complex128)


def random_state(n: int, rng: np.random.Generator, dtype: Any = np.complex128) -> npt.NDArray[Any]:
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    out: npt.NDArray[Any] = (v / np.linalg.norm(v)).astype(dtype)
    return out


def random_hermitian(dim: int, rng: np.random.Generator) -> npt.NDArray[np.complex128]:
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    h: npt.NDArray[np.complex128] = ((a + a.conj().T) / 2).astype(np.complex128)
    return h


@st.composite
def pauli_words(draw: st.DrawFn, max_n: int = 7) -> tuple[int, dict[int, str]]:
    n = draw(st.integers(min_value=1, max_value=max_n))
    letters = draw(st.lists(st.sampled_from("IXYZ"), min_size=n, max_size=n))
    return n, dict(enumerate(letters))


# --------------------------------------------------------------------------- Pauli fast path


@settings(max_examples=200, deadline=None)
@given(case=pauli_words(), seed=st.integers(0, 2**32 - 1), dtype=st.sampled_from(DTYPES))
def test_pauli_fast_path_equals_dense_kronecker_reference(
    case: tuple[int, dict[int, str]], seed: int, dtype: Any
) -> None:
    n, word = case
    psi = random_state(n, np.random.default_rng(seed), dtype)
    out = Pauli(word).apply(psi)
    assert out.dtype == np.dtype(dtype)  # no promotion (ADR-0001 D3)
    assert_close(out, dense_pauli(word, n) @ psi.astype(np.complex128), dtype)


@pytest.mark.parametrize("letter", "XYZ")
@pytest.mark.parametrize("wire", [0, 1, 2])
def test_single_paulis_on_each_wire(letter: str, wire: int) -> None:
    psi = random_state(3, np.random.default_rng(wire))
    obs = {"X": X, "Y": Y, "Z": Z}[letter](wire)
    assert_close(obs.apply(psi), dense_pauli({wire: letter}, 3) @ psi, np.complex128)


def test_identity_returns_a_copy() -> None:
    psi = random_state(2, np.random.default_rng(0))
    out = I().apply(psi)
    np.testing.assert_array_equal(out, psi)
    assert out is not psi


def test_apply_is_pure() -> None:
    psi = random_state(3, np.random.default_rng(1))
    before = psi.copy()
    (0.5 * Z(0) @ Y(2) + Hermitian(np.diag([1.0, 2.0]), [1])).apply(psi)
    np.testing.assert_array_equal(psi, before)


# --------------------------------------------------------------------------- algebra


def test_tensor_of_paulis_simplifies_to_one_pauli() -> None:
    obs = Z(0) @ X(2) @ Y(1)
    assert isinstance(obs, Pauli)
    assert obs.word == ((0, "Z"), (1, "Y"), (2, "X"))
    assert repr(obs) == "Z(0) @ Y(1) @ X(2)"


def test_scalar_multiplication_and_sums() -> None:
    h = 0.5 * Z(0) @ Z(1) - 0.2 * X(0)
    assert isinstance(h, Sum)
    assert h.terms == ((0.5, Z(0) @ Z(1)), (-0.2, X(0)))
    assert repr(h) == "Sum(0.5 * Z(0) @ Z(1), -0.2 * X(0))"
    assert (Z(0) * 2).terms == ((2.0, Z(0)),)


def test_nested_sums_flatten_and_scale() -> None:
    h = 2 * (Z(0) + 3 * X(1)) - Y(0)
    assert h.terms == ((2.0, Z(0)), (6.0, X(1)), (-1.0, Y(0)))
    assert (-Z(0)).terms == ((-1.0, Z(0)),)


def test_tensor_distributes_over_sums() -> None:
    h = (Z(0) + 2 * X(0)) @ Z(1)
    assert isinstance(h, Sum)
    assert h.terms == ((1.0, Z(0) @ Z(1)), (2.0, X(0) @ Z(1)))
    psi = random_state(2, np.random.default_rng(4))
    expected = (dense_pauli({0: "Z", 1: "Z"}, 2) + 2 * dense_pauli({0: "X", 1: "Z"}, 2)) @ psi
    assert_close(h.apply(psi), expected, np.complex128)


def test_tensor_distributes_over_a_sum_on_the_right() -> None:
    h = X(0) @ (Z(1) - 0.5 * Y(1))
    assert isinstance(h, Sum)
    assert h.terms == ((1.0, X(0) @ Z(1)), (-0.5, X(0) @ Y(1)))
    psi = random_state(2, np.random.default_rng(9))
    expected = (dense_pauli({0: "X", 1: "Z"}, 2) - 0.5 * dense_pauli({0: "X", 1: "Y"}, 2)) @ psi
    assert_close(h.apply(psi), expected, np.complex128)


def test_mixed_products_keep_pauli_and_hermitian_factors() -> None:
    m = random_hermitian(2, np.random.default_rng(5))
    obs = Hermitian(m, [1]) @ X(0) @ Z(2)
    assert isinstance(obs, Prod)
    assert obs.wires == (0, 1, 2)
    assert obs.factors[0] == X(0) @ Z(2)
    expected = np.kron(np.kron(_P["X"], m), _P["Z"])
    psi = random_state(3, np.random.default_rng(6))
    assert_close(obs.apply(psi), expected @ psi, np.complex128)


@pytest.mark.parametrize("bad", [Z(0) @ Z(1), Hermitian(np.eye(2), [1])])
def test_tensor_products_on_overlapping_wires_are_rejected(bad: Observable) -> None:
    with pytest.raises(ValueError, match="overlap on wire"):
        Z(1) @ bad


def test_arithmetic_with_unsupported_types_is_rejected() -> None:
    with pytest.raises(TypeError):
        Z(0) * "2"
    with pytest.raises(TypeError):
        Z(0) * True
    with pytest.raises(TypeError):
        Z(0) + 1  # type: ignore[operator]
    with pytest.raises(TypeError):
        Z(0) - 1  # type: ignore[operator]
    with pytest.raises(TypeError):
        Z(0) @ 1  # type: ignore[operator]


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("word", "error", "message"),
    [
        ({0: "Q"}, ValueError, "letters must be I, X, Y or Z"),
        ({-1: "X"}, ValueError, "non-negative"),
        ({1.5: "X"}, TypeError, "must be ints"),
        ([(0, "X"), (0, "Z")], ValueError, "appears twice"),
    ],
)
def test_pauli_validation(word: Any, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        Pauli(word)


def test_pauli_identity_letters_are_dropped() -> None:
    assert Pauli({0: "I", 1: "Z"}) == Z(1)
    assert repr(Pauli({0: "I"})) == "I()"


@pytest.mark.parametrize(
    ("matrix", "wires", "message"),
    [
        (np.eye(2), [], "at least one wire"),
        (np.eye(4), [0, 0], "distinct"),
        (np.eye(4), [0], "needs a 2x2 matrix"),
        (np.array([[0, 1], [0, 0]]), [0], "not Hermitian"),
        (np.array([[1, 1j], [1j, 1]]), [0], "not Hermitian"),
    ],
)
def test_hermitian_validation(matrix: Any, wires: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Hermitian(matrix, wires)


def test_hermitian_matrix_is_read_only_and_hashable() -> None:
    h = Hermitian(np.diag([1.0, -1.0]), [0])
    with pytest.raises(ValueError, match="read-only"):
        h.base_matrix[0, 0] = 5
    assert h == Hermitian(np.diag([1.0, -1.0]), [0])
    assert {h, Hermitian(np.diag([1.0, -1.0]), [0])} == {h}
    assert h != Hermitian(np.diag([1.0, -1.0]), [1])
    assert repr(Hermitian(np.eye(4), [1, 0])) == "Hermitian(<4x4>, wires=(1, 0))"


def test_hermitian_reports_sorted_and_target_wires() -> None:
    h = Hermitian(np.eye(4), [2, 0])
    assert h.wires == (0, 2)
    assert h.target_wires == (2, 0)
    assert repr(h @ X(1)) == "X(1) @ Hermitian(<4x4>, wires=(2, 0))"


def test_hermitian_wire_order_matters() -> None:
    m = random_hermitian(4, np.random.default_rng(7))
    psi = random_state(2, np.random.default_rng(8))
    swap = np.eye(4)[[0, 2, 1, 3]]
    assert_close(Hermitian(m, [1, 0]).apply(psi), swap @ m @ swap @ psi, np.complex128)


@pytest.mark.parametrize(
    ("coeff", "error"), [(np.nan, ValueError), (1j, TypeError), ("1", TypeError)]
)
def test_sum_coefficients_must_be_real_and_finite(coeff: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        Sum([(coeff, Z(0))])


def test_sum_validation() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        Sum([])
    with pytest.raises(TypeError, match="must be observables"):
        Sum([(1.0, "Z0")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Use `@`"):
        Prod([Z(0) + X(0), Z(1)])


def test_observable_outside_the_state_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Z\(3\) acts on wire\(s\) \[3\], but the state has 2"):
        Z(3).apply(np.array([1, 0, 0, 0], dtype=complex))


def test_equality_and_hashing() -> None:
    assert Z(0) @ Z(1) == Pauli({1: "Z", 0: "Z"})
    assert hash(Z(0) @ Z(1)) == hash(Pauli({1: "Z", 0: "Z"}))
    assert Z(0) != X(0)
    assert Z(0) != "Z(0)"
    assert (Z(0) + X(1)) == (Z(0) + X(1))
    assert len({Z(0) + X(1), Z(0) + X(1)}) == 1
    p = Hermitian(np.eye(2), [0]) @ Z(1)
    assert p == Hermitian(np.eye(2), [0]) @ Z(1)
    assert len({p, Hermitian(np.eye(2), [0]) @ Z(1)}) == 1


# --------------------------------------------------------------------------- dense matrices


@settings(max_examples=50, deadline=None)
@given(case=pauli_words(max_n=4))
def test_matrix_of_every_observable_is_hermitian(case: tuple[int, dict[int, str]]) -> None:
    n, word = case
    rng = np.random.default_rng(n)
    obs = 0.3 * Pauli(word) + Hermitian(random_hermitian(2, rng), [n - 1]) - 1.5 * I()
    m = obs.matrix(n)
    assert_close(m, m.conj().T, np.complex128)
