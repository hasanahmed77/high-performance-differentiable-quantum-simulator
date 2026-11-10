"""Shared numerical tolerances (plan.md §6.3).

Every floating-point comparison in the test suite uses these values. Don't loosen a tolerance
inline; if a looser bound is genuinely needed, justify it in an ADR.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Absolute and relative tolerance for one precision."""

    atol: float
    rtol: float


TOLERANCES: dict[np.dtype[Any], Tolerance] = {
    np.dtype(np.complex128): Tolerance(atol=1e-10, rtol=1e-8),
    np.dtype(np.complex64): Tolerance(atol=1e-5, rtol=1e-4),
}

GRADIENT_FD = Tolerance(atol=1e-5, rtol=0.0)
"""Gradients compared against central finite differences (step 1e-7, complex128)."""

GRADIENT_FD_COMPLEX64 = Tolerance(atol=1e-3, rtol=0.0)
"""Gradients compared against central finite differences in complex64 (step 1e-3).

Rounding error is about eps32 * |O| / h = 6e-5 * |O| (ADR-0004 D7); 2.1e-4 was the worst case
measured on 200 random circuits.
"""


def tolerance_for(dtype: npt.DTypeLike) -> Tolerance:
    """Return the tolerance for ``dtype`` (complex128 or complex64)."""
    return TOLERANCES[np.dtype(dtype)]


def assert_close(actual: npt.ArrayLike, expected: npt.ArrayLike, dtype: npt.DTypeLike) -> None:
    """Assert ``actual ≈ expected`` within the tolerance for ``dtype``."""
    tol = tolerance_for(dtype)
    actual_arr: npt.NDArray[Any] = np.asarray(actual)
    expected_arr: npt.NDArray[Any] = np.asarray(expected)
    np.testing.assert_allclose(actual_arr, expected_arr, atol=tol.atol, rtol=tol.rtol)
