"""Unit tests for the gradient front end and finite differences (QS-401, ADR-0004 D1, D2, D7)."""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.gradients import grad, jacobian
from ketjet.observables import X, Y, Z
from ketjet.operations import Parameter
from tests.tolerances import GRADIENT_FD, GRADIENT_FD_COMPLEX64

a, b, t = Parameter("a"), Parameter("b"), Parameter("t")
ANGLES = [-2.5, -0.7, 0.0, 0.4, 1.3, 3.0]


def assert_fd_close(actual: Any, expected: Any, tol: Any = GRADIENT_FD) -> None:
    np.testing.assert_allclose(actual, expected, atol=tol.atol, rtol=tol.rtol)


# --------------------------------------------------------------------------- closed forms


@pytest.mark.parametrize("x", ANGLES)
def test_single_rotation_matches_closed_form(x: float) -> None:
    c = Circuit(1).ry(t, 0)
    assert_fd_close(grad(c, Z(0), {"t": x}, method="finite-diff"), [-np.sin(x)])  # ⟨Z⟩ = cos t
    assert_fd_close(grad(c, X(0), {"t": x}, method="finite-diff"), [np.cos(x)])  # ⟨X⟩ = sin t


@pytest.mark.parametrize("x", ANGLES)
def test_phase_shift_matches_closed_form(x: float) -> None:
    c = Circuit(1).h(0).phase(t, 0)
    assert_fd_close(grad(c, X(0), {"t": x}, method="finite-diff"), [-np.sin(x)])
    assert_fd_close(grad(c, Y(0), {"t": x}, method="finite-diff"), [np.cos(x)])


@pytest.mark.parametrize("x", ANGLES)
def test_controlled_rotation_depends_on_the_control(x: float) -> None:
    on = Circuit(2).x(0).crx(t, 0, 1)
    off = Circuit(2).crx(t, 0, 1)
    assert_fd_close(grad(on, Z(1), {"t": x}, method="finite-diff"), [-np.sin(x)])
    assert grad(off, Z(1), {"t": x}, method="finite-diff").tolist() == [0.0]


@pytest.mark.parametrize("x", ANGLES)
def test_variance_gradient(x: float) -> None:
    # Var(Z) = 1 - cos²t = sin²t, so d/dt = sin 2t.
    c = Circuit(1).ry(t, 0)
    g = grad(c, Z(0), {"t": x}, method="finite-diff", measurement="var")
    assert_fd_close(g, [np.sin(2 * x)])


# --------------------------------------------------------------------------- symbols and slots


@pytest.mark.parametrize("x", ANGLES)
def test_shared_symbol_sums_its_slots(x: float) -> None:
    # Two RX(t) gates are RX(2t): ⟨Z⟩ = cos 2t, so d/dt = -2 sin 2t (chain rule over both slots).
    c = Circuit(1).rx(t, 0).rx(t, 0)
    assert_fd_close(grad(c, Z(0), {"t": x}, method="finite-diff"), [-2 * np.sin(2 * x)])


def test_jacobian_shape_and_parameter_order() -> None:
    # Parameters are ordered by first appearance: b, then a.
    c = Circuit(2).ry(b, 1).rx(a, 0)
    assert c.parameters == (b, a)
    jac = jacobian(c, [Z(0), Z(1), Z(0) @ Z(1)], {"a": 0.5, "b": 1.0}, method="finite-diff")
    assert jac.shape == (3, 2)
    assert jac.dtype == np.float64
    expected = [
        [0.0, -np.sin(0.5)],
        [-np.sin(1.0), 0.0],
        [-np.sin(1.0) * np.cos(0.5), -np.cos(1.0) * np.sin(0.5)],
    ]
    assert_fd_close(jac, expected)


def test_grad_is_the_first_jacobian_row() -> None:
    c = Circuit(2).ry(a, 0).cnot(0, 1).rx(b, 1)
    values = {"a": 0.3, "b": -1.1}
    obs = Z(0) @ Z(1)
    np.testing.assert_array_equal(
        grad(c, obs, values, method="finite-diff"),
        jacobian(c, [obs], values, method="finite-diff")[0],
    )


def test_values_keyed_by_parameter_or_name_agree() -> None:
    c = Circuit(1).rx(a, 0).ry(b, 0)
    by_name = grad(c, Z(0), {"a": 0.2, "b": 0.9}, method="finite-diff")
    by_symbol = grad(c, Z(0), {a: 0.2, b: 0.9}, method="finite-diff")
    np.testing.assert_array_equal(by_name, by_symbol)


def test_circuit_without_symbols_has_an_empty_gradient() -> None:
    jac = jacobian(Circuit(1).h(0), [Z(0), X(0)], {}, method="finite-diff")
    assert jac.shape == (2, 0)


def test_circuit_gradient_convenience_method() -> None:
    c = Circuit(1).ry(t, 0)
    np.testing.assert_array_equal(
        c.gradient(Z(0), {"t": 0.4}), grad(c, Z(0), {"t": 0.4}, method="finite-diff")
    )


def test_inputs_are_not_modified() -> None:
    c = Circuit(1).ry(t, 0)
    before = c.copy()
    values = {"t": 0.4}
    grad(c, Z(0), values, method="finite-diff")
    assert c == before
    assert values == {"t": 0.4}


# --------------------------------------------------------------------------- precision


@pytest.mark.parametrize("seed", range(5))
def test_complex64_uses_a_larger_default_step(seed: int) -> None:
    rng = np.random.default_rng(seed)
    va, vb = rng.uniform(-3, 3, 2)
    c = Circuit(3).h(0).ry(a, 0).cnot(0, 1).rx(b, 1).crz(a, 1, 2).h(2)
    obs = 0.7 * Z(0) @ Z(1) + 0.3 * X(2)
    values = {"a": va, "b": vb}
    reference = grad(c, obs, values, method="finite-diff")
    single = grad(c, obs, values, method="finite-diff", dtype="complex64")
    assert_fd_close(single, reference, GRADIENT_FD_COMPLEX64)
    # With the complex128 step, float32 rounding dominates and the estimate is useless.
    bad = grad(c, obs, values, method="finite-diff", dtype="complex64", step=1e-7)
    assert np.abs(bad - reference).max() > GRADIENT_FD_COMPLEX64.atol


def test_explicit_step_is_used() -> None:
    # With a huge step the central difference of cos is sin(x) * sin(h) / h, not sin(x).
    c = Circuit(1).ry(t, 0)
    g = grad(c, Z(0), {"t": 1.0}, method="finite-diff", step=1.0)
    np.testing.assert_allclose(g, [-np.sin(1.0) * np.sin(1.0)], atol=1e-12)


# --------------------------------------------------------------------------- errors


def test_missing_value_raises() -> None:
    with pytest.raises(ValueError, match=r"Missing value\(s\) for parameter\(s\): b"):
        grad(Circuit(1).rx(a, 0).ry(b, 0), Z(0), {"a": 0.1}, method="finite-diff")


def test_unknown_value_raises() -> None:
    with pytest.raises(ValueError, match="Unknown parameter"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": 0.1, "zz": 1.0}, method="finite-diff")


def test_non_finite_value_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": float("nan")}, method="finite-diff")


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method 'backprop'"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": 0.1}, method="backprop")  # type: ignore[arg-type]


def test_unknown_measurement_raises() -> None:
    with pytest.raises(ValueError, match="Unknown measurement 'probs'"):
        grad(
            Circuit(1).rx(a, 0),
            Z(0),
            {"a": 0.1},
            method="finite-diff",
            measurement="probs",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("step", [0.0, -1e-7, float("nan")])
def test_non_positive_step_raises(step: float) -> None:
    with pytest.raises(ValueError, match="step must be positive"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": 0.1}, method="finite-diff", step=step)
