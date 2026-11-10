"""Unit tests for vector-Jacobian products (QS-406, ADR-0004 D2, D6)."""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.backends import NumPyBackend
from ketjet.gradients import Method, jacobian, vjp
from ketjet.observables import Hermitian, Observable, X, Y, Z
from ketjet.operations import Parameter
from tests.random_circuits import random_circuit, symbolize
from tests.tolerances import GRADIENT_FD, GRADIENT_FD_COMPLEX64, assert_close

a, b = Parameter("a"), Parameter("b")
METHODS: list[Method] = ["adjoint", "parameter-shift", "finite-diff"]


def _observables(n_wires: int, rng: np.random.Generator) -> list[Observable]:
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    w = [int(x) for x in rng.permutation(n_wires)]
    return [
        Z(w[0]),
        0.4 * Z(w[0]) @ Z(w[1]) - 1.3 * X(w[1]),
        Hermitian(m + m.conj().T, wires=(w[0], w[1])),
        Y(w[1]) @ X(w[0]),
    ]


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(10))
def test_vjp_equals_contracted_jacobian(seed: int, method: Method, dtype: Any) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 5))
    ops_, values, _ = symbolize(random_circuit(n_wires, depth=15, rng=rng), rng)
    circuit = Circuit(n_wires, ops_)
    observables = _observables(n_wires, rng)
    cotangent = rng.normal(size=len(observables))
    expected = jacobian(circuit, observables, values, method=method, dtype=dtype).T @ cotangent
    result = vjp(circuit, observables, values, cotangent, method=method, dtype=dtype)
    assert result.shape == (len(circuit.parameters),)
    assert result.dtype == np.float64
    if method == "finite-diff":
        # Contracting differences of M outputs and differencing one combined output round
        # differently; both are within the finite-difference tolerance of each other.
        tol = GRADIENT_FD_COMPLEX64 if dtype == np.complex64 else GRADIENT_FD
        np.testing.assert_allclose(result, expected, atol=tol.atol, rtol=tol.rtol)
    else:
        assert_close(result, expected, dtype)


def test_vjp_closed_form() -> None:
    # ⟨Z0⟩ = cos a, ⟨Z1⟩ = cos b: v·J = (-v0 sin a, -v1 sin b).
    c = Circuit(2).rx(a, 0).ry(b, 1)
    result = vjp(c, [Z(0), Z(1)], {"a": 0.5, "b": 1.0}, [2.0, -1.0])
    np.testing.assert_allclose(result, [-2 * np.sin(0.5), np.sin(1.0)], atol=1e-12)


def test_unit_cotangent_selects_a_row() -> None:
    rng = np.random.default_rng(3)
    ops_, values, _ = symbolize(random_circuit(3, depth=10, rng=rng), rng)
    circuit = Circuit(3, ops_)
    observables = _observables(3, rng)
    jac = jacobian(circuit, observables, values)
    for m in range(len(observables)):
        e = np.eye(len(observables))[m]
        assert_close(vjp(circuit, observables, values, e), jac[m], np.complex128)


def test_vjp_accepts_integer_cotangents_and_sequences() -> None:
    c = Circuit(1).rx(a, 0)
    np.testing.assert_array_equal(
        vjp(c, [Z(0)], {"a": 0.3}, [2]), vjp(c, [Z(0)], {"a": 0.3}, np.array([2.0]))
    )


def test_vjp_without_observables_or_parameters() -> None:
    c = Circuit(1).rx(a, 0)
    np.testing.assert_array_equal(vjp(c, [], {"a": 0.3}, []), [0.0])
    assert vjp(Circuit(1).h(0), [Z(0), X(0)], {}, [1.0, 2.0]).shape == (0,)


def test_adjoint_vjp_makes_one_backward_pass_with_one_observable() -> None:
    """ADR-0004 D6: one backward pass over Σ v_m O_m, whatever the number of observables."""
    seen: list[int] = []

    class Counting(NumPyBackend):
        def adjoint_jacobian(self, *args: Any, **kwargs: Any) -> Any:
            seen.append(len(args[2]))  # observables
            return super().adjoint_jacobian(*args, **kwargs)

    c = Circuit(3).ry(a, 0).cnot(0, 1).crx(b, 1, 2)
    observables = [Z(0), Z(1), Z(2), X(0) @ X(1), Y(2)]
    vjp(c, observables, {"a": 0.1, "b": 0.2}, np.arange(5.0), backend=Counting())
    assert seen == [1]


def test_vjp_supports_variance_with_finite_differences() -> None:
    # Var(Z) = sin² a for RX(a)|0⟩, so d/da = sin 2a.
    c = Circuit(1).rx(a, 0)
    result = vjp(c, [Z(0)], {"a": 0.4}, [3.0], method="finite-diff", measurement="var")
    np.testing.assert_allclose(result, [3 * np.sin(0.8)], atol=GRADIENT_FD.atol)


def test_vjp_with_shots_is_reproducible() -> None:
    c = Circuit(1).rx(a, 0)
    kwargs: Any = {"method": "parameter-shift", "shots": 1000, "seed": 5}
    first = vjp(c, [Z(0), X(0)], {"a": 0.4}, [1.0, 2.0], **kwargs)
    np.testing.assert_array_equal(first, vjp(c, [Z(0), X(0)], {"a": 0.4}, [1.0, 2.0], **kwargs))


@pytest.mark.parametrize(
    ("cotangent", "error", "match"),
    [
        ([1.0], ValueError, r"must have shape \(2,\), one entry per observable, got \(1,\)"),
        ([[1.0, 2.0]], ValueError, r"must have shape .*got \(1, 2\)"),
        (1.0, ValueError, r"must have shape .*got \(\)"),
        ([1.0, np.nan], ValueError, "must be finite"),
        ([1.0, np.inf], ValueError, "must be finite"),
        ([1.0, 1j], TypeError, "must be real, got dtype complex128"),
        (["x", "y"], TypeError, "must be real"),
    ],
)
@pytest.mark.parametrize("method", METHODS)
def test_vjp_rejects_bad_cotangents(
    cotangent: Any, error: type[Exception], match: str, method: Method
) -> None:
    c = Circuit(1).rx(a, 0)
    with pytest.raises(error, match=f"The cotangent {match}"):
        vjp(c, [Z(0), X(0)], {"a": 0.1}, cotangent, method=method)


def test_vjp_reports_option_errors_like_jacobian() -> None:
    c = Circuit(1).rx(a, 0)
    with pytest.raises(ValueError, match="Unknown method"):
        vjp(c, [Z(0)], {"a": 0.1}, [1.0], method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="use method='parameter-shift'"):
        vjp(c, [Z(0)], {"a": 0.1}, [1.0], shots=10)
    with pytest.raises(ValueError, match="use method='finite-diff'"):
        vjp(c, [Z(0)], {"a": 0.1}, [1.0], measurement="var")
    with pytest.raises(ValueError, match="Missing value"):
        vjp(c, [Z(0)], {}, [1.0])
