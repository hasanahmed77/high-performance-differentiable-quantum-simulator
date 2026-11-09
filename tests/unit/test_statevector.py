"""Tests for statevector allocation, precision handling and memory accounting (QS-202, ADR-0001)."""

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ketjet import statevector as sv
from tests.tolerances import TOLERANCES, assert_close, tolerance_for

DTYPES = [np.complex128, np.complex64]


# --------------------------------------------------------------------------- dtype handling


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (np.complex128, np.complex128),
        (np.complex64, np.complex64),
        ("complex128", np.complex128),
        ("complex64", np.complex64),
        (np.dtype("complex64"), np.complex64),
    ],
)
def test_as_complex_dtype_accepts_supported_spellings(spec: Any, expected: Any) -> None:
    assert sv.as_complex_dtype(spec) == np.dtype(expected)


@pytest.mark.parametrize(
    "spec",
    [
        np.float64,
        np.complex256 if hasattr(np, "complex256") else "c32",
        "int32",
        "not-a-dtype",
        object,
    ],
)
def test_as_complex_dtype_rejects_unsupported(spec: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        sv.as_complex_dtype(spec)


# --------------------------------------------------------------------------- n_wires validation


@pytest.mark.parametrize("n", [1, 2, 10, sv.MAX_QUBITS, np.int64(5)])
def test_validate_n_wires_accepts_valid(n: int) -> None:
    assert sv.validate_n_wires(n) == int(n)
    assert type(sv.validate_n_wires(n)) is int


@pytest.mark.parametrize("n", [0, -1, sv.MAX_QUBITS + 1])
def test_validate_n_wires_rejects_out_of_range(n: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 63"):
        sv.validate_n_wires(n)


@pytest.mark.parametrize("n", [2.0, "3", None, True])
def test_validate_n_wires_rejects_non_integers(n: Any) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        sv.validate_n_wires(n)


# --------------------------------------------------------------------------- memory accounting


@pytest.mark.parametrize(("dtype", "itemsize"), [(np.complex128, 16), (np.complex64, 8)])
@pytest.mark.parametrize("n", [1, 5, 20, sv.MAX_QUBITS])
def test_state_nbytes_is_exact(n: int, dtype: Any, itemsize: int) -> None:
    assert sv.state_nbytes(n, dtype) == 2**n * itemsize


def test_state_nbytes_matches_actual_allocation() -> None:
    for dtype in DTYPES:
        assert sv.zero_state(12, dtype).nbytes == sv.state_nbytes(12, dtype)


def test_thirty_qubits_complex64_is_eight_gib() -> None:
    assert sv.state_nbytes(30, "complex64") == 8 * 2**30


# --------------------------------------------------------------------------- zero_state


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_zero_state_layout(n: int, dtype: Any) -> None:
    state = sv.zero_state(n, dtype)
    expected = np.zeros(2**n, dtype=dtype)
    expected[0] = 1
    assert state.shape == (2**n,)
    assert state.dtype == np.dtype(dtype)
    assert state.flags.c_contiguous
    np.testing.assert_array_equal(state, expected)


def test_zero_state_defaults_to_complex128() -> None:
    assert sv.zero_state(3).dtype == np.complex128


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_state_is_normalized(dtype: Any) -> None:
    assert_close(np.linalg.norm(sv.zero_state(8, dtype)), 1.0, dtype)


def test_zero_state_returns_independent_arrays() -> None:
    a, b = sv.zero_state(2), sv.zero_state(2)
    a[1] = 5
    assert b[1] == 0


def test_zero_state_respects_max_bytes_budget() -> None:
    assert sv.zero_state(4, max_bytes=sv.state_nbytes(4)).size == 16
    with pytest.raises(MemoryError, match=r"5-qubit complex128 state needs 512 B.*max_bytes=256 B"):
        sv.zero_state(5, max_bytes=256)


def test_max_bytes_error_uses_readable_units() -> None:
    with pytest.raises(MemoryError, match=r"needs 8\.0 GiB, which exceeds max_bytes=4\.0 GiB"):
        sv.zero_state(30, np.complex64, max_bytes=4 * 2**30)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "1"])
def test_max_bytes_must_be_positive_int(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_bytes must be a positive int"):
        sv.zero_state(2, max_bytes=bad)


def test_allocation_failure_becomes_clear_memory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_zeros(*args: Any, **kwargs: Any) -> Any:
        raise MemoryError

    monkeypatch.setattr(np, "zeros", failing_zeros)
    with pytest.raises(
        MemoryError, match=r"Could not allocate a 40-qubit complex64 state \(8\.0 TiB\)"
    ):
        sv.zero_state(40, "complex64")


# --------------------------------------------------------------------------- properties


@given(n=st.integers(min_value=1, max_value=14), dtype=st.sampled_from(DTYPES))
def test_zero_state_properties(n: int, dtype: Any) -> None:
    state = sv.zero_state(n, dtype)
    assert state.size == 2**n
    assert np.count_nonzero(state) == 1
    assert state[0] == 1
    assert state.nbytes == sv.state_nbytes(n, dtype)


# --------------------------------------------------------------------------- tolerances module


def test_tolerances_cover_every_supported_dtype() -> None:
    assert set(TOLERANCES) == set(sv.SUPPORTED_DTYPES)
    assert tolerance_for(np.complex64).atol > tolerance_for(np.complex128).atol
