"""Finite-difference gradients equal PennyLane's backprop gradients (QS-401, ADR-0004).

PennyLane differentiates its own native gates by automatic differentiation, which is independent of
both our gates and our finite differences. Every random-circuit parameter becomes its own symbol,
except that some are deliberately shared, to exercise the chain rule over slots.
"""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.gradients import jacobian
from ketjet.observables import Hermitian, Observable, X, Y, Z
from ketjet.operations import Operation, Parameter
from tests.tolerances import GRADIENT_FD

qml = pytest.importorskip("pennylane")
pytestmark = [
    pytest.mark.pennylane,
    # autograd's own complex-to-real cast inside PennyLane's backprop; not our code.
    pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning"),
]

from tests.pennylane_support import (  # noqa: E402
    random_circuit,
    to_pennylane,
    to_pennylane_observable,
)


def _symbolize(
    circuit: list[Operation], rng: np.random.Generator
) -> tuple[list[Operation], dict[str, float], list[int]]:
    """Replace every parameter by a symbol; about a quarter of slots reuse an earlier symbol.

    Returns the symbolic operations, the symbol values, and for each slot (in order) the index of
    its symbol in first-appearance order.
    """
    names: list[str] = []
    values: dict[str, float] = {}
    slot_symbol: list[int] = []
    symbolic = []
    for op in circuit:
        params = []
        for p in op.params:
            if names and rng.random() < 0.25:
                k = int(rng.integers(len(names)))
            else:
                k = len(names)
                names.append(f"p{k}")
                values[f"p{k}"] = float(p)  # type: ignore[arg-type]  # numeric before symbolizing
            slot_symbol.append(k)
            params.append(Parameter(names[k]))
        symbolic.append(Operation(op.gate, op.wires, tuple(params), op.controls, op.control_values))
    return symbolic, values, slot_symbol


def _random_observables(n_wires: int, rng: np.random.Generator) -> list[Observable]:
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    w = [int(x) for x in rng.permutation(n_wires)]
    return [
        Z(w[0]),
        X(w[0]) @ Y(w[1]),
        0.4 * Z(w[0]) @ Z(w[1]) - 1.3 * X(w[1]),
        Hermitian(m + m.conj().T, wires=(w[1],)) @ Z(w[0]),
    ]


@pytest.mark.parametrize("seed", range(15))
def test_finite_diff_matches_pennylane_backprop(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n_wires = int(rng.integers(2, 5))
    ops, values, slot_symbol = _symbolize(random_circuit(n_wires, depth=12, rng=rng), rng)
    circuit = Circuit(n_wires, ops)
    observables = _random_observables(n_wires, rng)
    ours = jacobian(circuit, observables, values, method="finite-diff")

    names = [p.name for p in circuit.parameters]
    theta = qml.numpy.array([values[n] for n in names], requires_grad=True)
    device = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(device, diff_method="backprop")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        slot = 0
        for op in ops:
            params = [x[slot_symbol[slot + j]] for j in range(len(op.params))]
            slot += len(op.params)
            to_pennylane(op, params)
        return [qml.expval(to_pennylane_observable(o)) for o in observables]

    if not names:  # pragma: no cover  # depends on the random draw
        assert ours.shape == (len(observables), 0)
        return
    theirs = qml.jacobian(lambda x: qml.math.stack(run(x)))(theta)
    np.testing.assert_allclose(ours, theirs, atol=GRADIENT_FD.atol, rtol=GRADIENT_FD.rtol)
