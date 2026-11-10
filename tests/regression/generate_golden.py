"""Regenerate the regression golden files (run deliberately, never from CI).

    uv run python -m tests.regression.generate_golden

Every value is cross-checked against PennyLane's default.qubit before it's written, so a golden
file can't capture a wrong answer. Regenerating changes pinned behaviour: explain why in the PR.
"""

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml

from ketjet import Circuit
from ketjet.gradients import jacobian
from ketjet.observables import Observable
from ketjet.operations import Parameter
from tests.pennylane_support import run_default_qubit, to_pennylane, to_pennylane_observable
from tests.regression.circuits import CIRCUITS, GRADIENT_CIRCUITS, observables

GOLDEN = Path(__file__).parent / "golden"


def pennylane_jacobian(
    circuit: Circuit, observables: list[Observable], values: dict[str, float]
) -> Any:
    """The Jacobian by PennyLane's backprop, independent of our gradient code."""
    names = [p.name for p in circuit.parameters]
    index = {name: k for k, name in enumerate(names)}

    @qml.qnode(qml.device("default.qubit", wires=circuit.n_wires), diff_method="backprop")  # type: ignore[untyped-decorator]
    def run(x: Any) -> Any:
        for op in circuit:
            params = [x[index[p.name]] if isinstance(p, Parameter) else p for p in op.params]
            to_pennylane(op, params)
        return qml.math.stack([qml.expval(to_pennylane_observable(o)) for o in observables])

    theta = qml.numpy.array([values[n] for n in names], requires_grad=True)
    return qml.jacobian(run)(theta)


def write_gradients() -> None:
    for name, (circuit, values) in GRADIENT_CIRCUITS.items():
        named = observables(circuit.n_wires)
        jac = jacobian(circuit, list(named.values()), values)
        expected = pennylane_jacobian(circuit, list(named.values()), values)
        np.testing.assert_allclose(jac, expected, atol=1e-12)
        record = {
            "circuit": circuit.to_dict(),
            "values": values,
            "parameters": [p.name for p in circuit.parameters],
            "jacobian": {key: row.tolist() for key, row in zip(named, jac, strict=True)},
        }
        (GOLDEN / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {name}.json (cross-checked against default.qubit backprop)")


def main() -> None:
    write_gradients()
    for name, circuit in CIRCUITS.items():
        n = circuit.n_wires
        ops = list(circuit)
        state = circuit.state()
        probs = circuit.probs()
        np.testing.assert_allclose(state, run_default_qubit(ops, n, qml.state), atol=1e-12)
        np.testing.assert_allclose(probs, run_default_qubit(ops, n, qml.probs), atol=1e-12)
        expvals: dict[str, float] = {}
        variances: dict[str, float] = {}
        for key, obs in observables(n).items():
            pl = to_pennylane_observable(obs)
            expvals[key] = circuit.expval(obs)
            variances[key] = circuit.var(obs)
            np.testing.assert_allclose(
                expvals[key], run_default_qubit(ops, n, partial(qml.expval, pl)), atol=1e-12
            )
            np.testing.assert_allclose(
                variances[key], run_default_qubit(ops, n, partial(qml.var, pl)), atol=1e-12
            )
        record = {
            "circuit": circuit.to_dict(),
            "state": {"real": state.real.tolist(), "imag": state.imag.tolist()},
            "probs": probs.tolist(),
            "expval": expvals,
            "var": variances,
        }
        (GOLDEN / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {name}.json (cross-checked against default.qubit)")


if __name__ == "__main__":
    main()
