"""Our gradients equal PennyLane's backprop gradients (QS-401, QS-402, ADR-0004).

PennyLane differentiates its own native gates by automatic differentiation, which is independent of
our gates, our shift rules and our finite differences. Exact methods must agree at the standard
complex128 tolerance; finite differences at the gradient tolerance. Every random-circuit
parameter becomes its own symbol, except that some are deliberately shared, to exercise the chain
rule over slots.
"""

from typing import Any

import numpy as np
import pytest

from ketjet import Circuit
from ketjet.gradients import Method, jacobian
from ketjet.observables import Hermitian, Observable, X, Y, Z
from tests.random_circuits import symbolize
from tests.tolerances import GRADIENT_FD, TOLERANCES, Tolerance

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


def _random_observables(n_wires: int, rng: np.random.Generator) -> list[Observable]:
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    w = [int(x) for x in rng.permutation(n_wires)]
    return [
        Z(w[0]),
        X(w[0]) @ Y(w[1]),
        0.4 * Z(w[0]) @ Z(w[1]) - 1.3 * X(w[1]),
        Hermitian(m + m.conj().T, wires=(w[1],)) @ Z(w[0]),
    ]


METHODS: dict[Method, Tolerance] = {
    "parameter-shift": TOLERANCES[np.dtype(np.complex128)],
    "finite-diff": GRADIENT_FD,
}


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("seed", range(15))
def test_gradients_match_pennylane_backprop(seed: int, method: Method) -> None:
    rng = np.random.default_rng(1000 + seed)
    n_wires = int(rng.integers(2, 5))
    ops, values, slot_symbol = symbolize(random_circuit(n_wires, depth=12, rng=rng), rng)
    circuit = Circuit(n_wires, ops)
    observables = _random_observables(n_wires, rng)
    ours = jacobian(circuit, observables, values, method=method)

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
    tol = METHODS[method]
    np.testing.assert_allclose(ours, theirs, atol=tol.atol, rtol=tol.rtol)
