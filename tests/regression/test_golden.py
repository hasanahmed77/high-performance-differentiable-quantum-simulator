"""Regression tests: canonical circuits keep producing their pinned, cross-checked outputs.

Golden files are produced by ``generate_golden.py``, which verifies every value against
PennyLane's default.qubit (for gradients, its backprop) before writing it. A failure here means a
change altered a known-good answer.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.gradients import Method, jacobian
from tests.regression.circuits import CIRCUITS, GRADIENT_CIRCUITS, observables
from tests.tolerances import GRADIENT_FD, GRADIENT_FD_COMPLEX64, assert_close

GOLDEN = Path(__file__).parent / "golden"


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((GOLDEN / f"{name}.json").read_text())
    return data


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_golden_circuit_definition_is_unchanged(name: str) -> None:
    assert Circuit.from_dict(load(name)["circuit"]) == CIRCUITS[name]


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_golden_outputs(name: str, dtype: Any) -> None:
    golden = load(name)
    circuit = Circuit.from_dict(golden["circuit"])
    state = np.array(golden["state"]["real"]) + 1j * np.array(golden["state"]["imag"])
    assert_close(circuit.state(dtype=dtype), state, dtype)
    assert_close(circuit.probs(dtype=dtype), golden["probs"], dtype)
    for key, obs in observables(circuit.n_wires).items():
        assert_close(circuit.expval(obs, dtype=dtype), golden["expval"][key], dtype)
        assert_close(circuit.var(obs, dtype=dtype), golden["var"][key], dtype)


@pytest.mark.parametrize(("name", "n", "j"), [("qft4_0110", 4, 0b0110), ("qft5_10011", 5, 0b10011)])
def test_qft_matches_the_analytic_formula(name: str, n: int, j: int) -> None:
    """QFT|j> = sum_k exp(2 pi i j k / N) |k> / sqrt(N): independent of any simulator."""
    size = 2**n
    expected = np.exp(2j * np.pi * j * np.arange(size) / size) / np.sqrt(size)
    assert_close(CIRCUITS[name].state(), expected, np.complex128)


def test_ghz_has_only_the_two_extreme_outcomes() -> None:
    p = CIRCUITS["ghz5"].probs()
    assert_close(p[[0, -1]], [0.5, 0.5], np.complex128)
    assert_close(p[1:-1], np.zeros(30), np.complex128)


@pytest.mark.parametrize("name", sorted(GRADIENT_CIRCUITS))
def test_golden_gradient_definition_is_unchanged(name: str) -> None:
    golden = load(name)
    circuit, values = GRADIENT_CIRCUITS[name]
    assert Circuit.from_dict(golden["circuit"]) == circuit
    assert golden["values"] == values
    assert golden["parameters"] == [p.name for p in circuit.parameters]


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("method", ["adjoint", "parameter-shift", "finite-diff"])
@pytest.mark.parametrize("name", sorted(GRADIENT_CIRCUITS))
def test_golden_gradient(name: str, method: Method, dtype: Any) -> None:
    """Every method reproduces the pinned Jacobian, cross-checked against PennyLane backprop."""
    golden = load(name)
    circuit = Circuit.from_dict(golden["circuit"])
    named = observables(circuit.n_wires)
    jac = jacobian(circuit, list(named.values()), golden["values"], method=method, dtype=dtype)
    expected = np.array([golden["jacobian"][key] for key in named])
    if method == "finite-diff":
        tol = GRADIENT_FD if dtype == np.complex128 else GRADIENT_FD_COMPLEX64
        np.testing.assert_allclose(jac, expected, atol=tol.atol, rtol=tol.rtol)
    else:
        assert_close(jac, expected, dtype)
