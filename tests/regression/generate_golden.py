"""Regenerate the regression golden files (run deliberately, never from CI).

    uv run python -m tests.regression.generate_golden

Every value is cross-checked against PennyLane's default.qubit before it's written, so a golden
file can't capture a wrong answer. Regenerating changes pinned behaviour: explain why in the PR.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pennylane as qml

from tests.pennylane_support import run_default_qubit, to_pennylane_observable
from tests.regression.circuits import CIRCUITS, observables

GOLDEN = Path(__file__).parent / "golden"


def main() -> None:
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
