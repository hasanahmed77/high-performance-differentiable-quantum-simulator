"""Unit tests for parameter-shift gradients (QS-402, ADR-0004 D5)."""

import zlib
from typing import Any

import numpy as np
import pytest

from ketjet import Circuit, measurements
from ketjet.gates import GATES, GateSpec, frequencies
from ketjet.gradients import grad, jacobian, shift_rule
from ketjet.kernels import apply_matrix
from ketjet.observables import Hermitian, X, Y, Z
from ketjet.operations import Operation, Parameter
from tests.random_circuits import random_circuit, symbolize
from tests.tolerances import GRADIENT_FD, TOLERANCES, assert_close

a, b, t = Parameter("a"), Parameter("b"), Parameter("t")
PARAMETRIZED = sorted(name for name, spec in GATES.items() if spec.num_params)
C128 = TOLERANCES[np.dtype(np.complex128)]


# --------------------------------------------------------------------------- spectra and rules


def test_frequencies_of_common_spectra() -> None:
    assert frequencies((-0.5, 0.5)) == (1.0,)
    assert frequencies((-0.5, 0.5), controlled=True) == (0.5, 1.0)
    assert frequencies((0.0, 1.0)) == frequencies((0.0, 1.0), controlled=True) == (1.0,)
    assert frequencies((0.3, 0.3)) == ()
    assert frequencies((0.0, 1.0, 2.0)) == (1.0, 2.0)


def test_non_equidistant_frequencies_are_rejected() -> None:
    assert frequencies((0.0, 1.0, 3.0)) == (1.0, 2.0, 3.0)  # gaps 1, 2, 3: equidistant
    with pytest.raises(ValueError, match="not equidistant"):
        frequencies((0.0, 1.0, 2.5))  # gaps {1, 1.5, 2.5}


def test_a_control_can_make_a_spectrum_invalid() -> None:
    # {1.5, 2.5} has the single gap 1; a control adds 0, giving gaps {1, 1.5, 2.5}.
    assert frequencies((1.5, 2.5)) == (1.0,)
    with pytest.raises(ValueError, match="not equidistant"):
        frequencies((1.5, 2.5), controlled=True)


def test_gate_spec_validates_its_spectra() -> None:
    m, d = GATES["RX"].matrix_fn, GATES["RX"].derivative_fn
    with pytest.raises(ValueError, match="declares 0 generator spectra"):
        GateSpec("Bad", 1, 1, m, lambda p: ("Bad", p), derivative_fn=d)
    with pytest.raises(ValueError, match="not equidistant"):
        GateSpec(
            "Bad", 1, 1, m, lambda p: ("Bad", p), generator_eigvals=((1.5, 2.5),), derivative_fn=d
        )


def test_every_parametrized_gate_declares_a_spectrum() -> None:
    for name in PARAMETRIZED:
        spec = GATES[name]
        assert len(spec.generator_eigvals) == spec.num_params
        for j in range(spec.num_params):
            assert spec.frequencies(j) == (1.0,)
            assert spec.frequencies(j, controlled=True) in {(1.0,), (0.5, 1.0)}


def test_two_term_rule() -> None:
    np.testing.assert_allclose(shift_rule((1.0,)), [(0.5, np.pi / 2)], rtol=1e-15)
    # A frequency Ω scales shifts by 1/Ω and coefficients by Ω.
    np.testing.assert_allclose(shift_rule((2.0,)), [(1.0, np.pi / 4)], rtol=1e-15)
    assert shift_rule(()) == ()


def test_four_term_rule_for_controlled_rotations() -> None:
    # The known constants (Anselmetti et al., 2021): c± = (√2 ± 1) / (4√2), shifts π/2 and 3π/2.
    root2 = np.sqrt(2)
    expected = [((root2 + 1) / (4 * root2), np.pi / 2), (-(root2 - 1) / (4 * root2), 3 * np.pi / 2)]
    np.testing.assert_allclose(shift_rule((0.5, 1.0)), expected, rtol=1e-14)


def _random_state(n: int, rng: np.random.Generator) -> Any:
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return psi / np.linalg.norm(psi)


def _random_hermitian(n: int, rng: np.random.Generator) -> Hermitian:
    m = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    return Hermitian(m + m.conj().T, wires=tuple(range(n)))


@pytest.mark.parametrize("n_controls", [0, 1, 2])
@pytest.mark.parametrize("name", PARAMETRIZED)
def test_declared_spectrum_is_exactly_the_fourier_spectrum(name: str, n_controls: int) -> None:
    """f(x) = ⟨ψ|U(x)† O U(x)|ψ⟩ has exactly the declared frequencies, and no others.

    Frequencies are multiples of 1/2, so f has period 4π; N samples over [0, 4π) put frequency k/2
    in FFT bin k.
    """
    rng = np.random.default_rng(zlib.crc32(f"{name}/{n_controls}".encode()))
    spec = GATES[name]
    n = 1 + n_controls
    psi, obs = _random_state(n, rng), _random_hermitian(n, rng)
    n_samples = 16
    grid = np.arange(n_samples) * 4 * np.pi / n_samples
    for j in range(spec.num_params):
        base = rng.uniform(-np.pi, np.pi, spec.num_params)

        def f(x: float, j: int = j, base: Any = base) -> float:
            params = tuple(float(x) if k == j else float(p) for k, p in enumerate(base))
            op = Operation(name, (n_controls,), params, tuple(range(n_controls)))
            return measurements.expval(apply_matrix(psi, op.matrix(), op.all_wires), obs)

        spectrum = np.abs(np.fft.rfft([f(x) for x in grid])) / n_samples
        declared = {round(2 * w) for w in spec.frequencies(j, controlled=n_controls > 0)}
        for k in range(1, len(spectrum)):
            if k in declared:
                assert spectrum[k] > 1e-3, f"{name}[{j}] lacks declared frequency {k / 2}"
            else:
                assert spectrum[k] < 1e-12, f"{name}[{j}] has undeclared frequency {k / 2}"


# --------------------------------------------------------------------------- gradients


@pytest.mark.parametrize("x", [-2.5, -0.7, 0.4, 1.3, 3.0])
def test_two_term_rule_is_wrong_for_a_controlled_rotation(x: float) -> None:
    """The textbook rule silently biases controlled rotations; the four-term rule is exact.

    The half frequency comes from interference between the control's branches, so the observable
    must act coherently on the control (an observable on the target alone hides the bias).
    """
    c = Circuit(2).h(0).ry(0.7, 1).crx(t, 0, 1)
    obs = X(0) @ Z(1) + 0.5 * Y(0) @ X(1)
    reference = grad(c, obs, {"t": x}, method="finite-diff")[0]

    def f(v: float) -> float:
        return c.bind(t=v).expval(obs)

    two_term = 0.5 * (f(x + np.pi / 2) - f(x - np.pi / 2))
    assert abs(two_term - reference) > 1e-2
    four_term = grad(c, obs, {"t": x}, method="parameter-shift")[0]
    assert four_term == pytest.approx(reference, abs=GRADIENT_FD.atol)


@pytest.mark.parametrize("seed", range(20))
def test_parameter_shift_matches_finite_differences(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_wires = int(rng.integers(2, 5))
    ops, values, _ = symbolize(random_circuit(n_wires, depth=15, rng=rng), rng)
    circuit = Circuit(n_wires, ops)
    observables = [Z(0), Z(0) @ X(1) - 0.3 * Z(1)]
    exact = jacobian(circuit, observables, values, method="parameter-shift")
    approx = jacobian(circuit, observables, values, method="finite-diff")
    np.testing.assert_allclose(exact, approx, atol=GRADIENT_FD.atol, rtol=GRADIENT_FD.rtol)


@pytest.mark.parametrize("seed", range(5))
def test_parameter_shift_in_complex64(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    ops, values, _ = symbolize(random_circuit(3, depth=15, rng=rng), rng)
    circuit = Circuit(3, ops)
    observables = [Z(0), X(1) @ Z(2)]
    double = jacobian(circuit, observables, values, method="parameter-shift")
    single = jacobian(circuit, observables, values, method="parameter-shift", dtype="complex64")
    assert_close(single, double, np.complex64)


def test_parameter_shift_rejects_variance() -> None:
    with pytest.raises(ValueError, match="can't differentiate 'var'; use method='finite-diff'"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": 0.1}, method="parameter-shift", measurement="var")


def test_finite_differences_reject_shots() -> None:
    with pytest.raises(ValueError, match="use method='parameter-shift'"):
        grad(Circuit(1).rx(a, 0), Z(0), {"a": 0.1}, method="finite-diff", shots=100)


# --------------------------------------------------------------------------- shots


SHOT_CIRCUIT = Circuit(2).ry(a, 0).cnot(0, 1).crx(b, 0, 1).ry(0.4, 1)
SHOT_VALUES = {"a": 0.9, "b": -1.4}


def test_shot_gradients_are_reproducible() -> None:
    first = grad(SHOT_CIRCUIT, Z(1), SHOT_VALUES, shots=500, seed=11, method="parameter-shift")
    again = grad(SHOT_CIRCUIT, Z(1), SHOT_VALUES, shots=500, seed=11, method="parameter-shift")
    other = grad(SHOT_CIRCUIT, Z(1), SHOT_VALUES, shots=500, seed=12, method="parameter-shift")
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    from_generator = grad(
        SHOT_CIRCUIT,
        Z(1),
        SHOT_VALUES,
        shots=500,
        seed=np.random.default_rng(11),
        method="parameter-shift",
    )
    np.testing.assert_array_equal(first, from_generator)


def test_every_shifted_evaluation_draws_independently() -> None:
    # Two symbols on identical gates: correlated draws would give identical estimates.
    c = Circuit(1).ry(a, 0).ry(b, 0)
    g = grad(c, Z(0), {"a": 0.5, "b": 0.5}, shots=200, seed=3, method="parameter-shift")
    assert g[0] != g[1]


@pytest.mark.parametrize("seed", range(10))
def test_shot_gradients_are_within_four_sigma(seed: int) -> None:
    """Each slot's estimate is Σ c [f̂(+s) - f̂(-s)], and every f̂ averages ±1 eigenvalues of Z."""
    shots = 2000
    estimate = grad(
        SHOT_CIRCUIT, Z(1), SHOT_VALUES, shots=shots, seed=seed, method="parameter-shift"
    )
    exact = grad(SHOT_CIRCUIT, Z(1), SHOT_VALUES)
    for p, (op_index, rule) in enumerate([(0, shift_rule((1.0,))), (2, shift_rule((0.5, 1.0)))]):
        name = SHOT_CIRCUIT.parameters[p].name
        assert SHOT_CIRCUIT.operations[op_index].params[0].name == name  # type: ignore[union-attr]
        variance = 0.0
        for coefficient, shift in rule:
            for sign in (1, -1):
                shifted = {**SHOT_VALUES, name: SHOT_VALUES[name] + sign * shift}
                mean = SHOT_CIRCUIT.bind(shifted).expval(Z(1))
                variance += coefficient**2 * (1 - mean**2) / shots
        assert abs(estimate[p] - exact[p]) <= 4 * np.sqrt(variance) + C128.atol
