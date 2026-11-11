# Sprint 04: PennyLane device and v0.1.0

| | |
|---|---|
| **Status** | In progress |
| **Started** | 2026-11-11 |
| **Milestone** | `Sprint 04 – PennyLane device` |
| **Sprint goal** | Anyone can `pip install "ketjet[pennylane]"` from PyPI and run, and differentiate, any PennyLane QNode on `qml.device("ketjet.qubit")`, with PennyLane's own device test suite passing. Release **v0.1.0**. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-501 | Spike and ADR-0005: the PennyLane device design | 2 | n/a | #126 |
| QS-502 | `qml.device("ketjet.qubit")` via an entry point; tapes translate 1:1 and execute; analytic and shot measurements | 5 | ADR-0005 | |
| QS-503 | Unsupported operations decompose automatically; clear errors for anything that can't | 3 | QS-502 | |
| QS-504 | Device gradients: `diff_method="adjoint"` and single-pass VJP (backprop deferred to the JAX backend, ADR-0005 D4) | 5 | QS-503 | |
| QS-505 | `pl-device-test` passing in CI, with skips documented | 3 | QS-504 | |
| QS-1203 | README for the release: problem, design, results, install, quickstart, badges | 2 | QS-505 | |
| QS-1205 | Release pipeline dry run (TestPyPI and smoke tests) and the real **v0.1.0** release | 5 | QS-1203 | |
| QS-1202 | First runnable examples: Bell, QFT and VQE scripts on `ketjet.qubit`, executed in CI (**stretch**) | 3 | QS-502 | |
| | **Total** | **28** (25 committed + 3 stretch) | | |

**Capacity.** 28 points is well above the mean throughput of 19.5, so 25 points are committed and QS-1202 is a
stretch goal. If it slips, v0.1.0 still ships with the README quickstart and the H₂ example, and QS-1202 moves to
Sprint 5. QS-1205 may come in under its estimate, since QS-111 already built and lint-checked the pipeline. What
remains is the first end-to-end run.

**Sequencing.** The device PRs are serial: each one's tests build on the previous one. Releasing comes last, so
v0.1.0 is cut from a `main` that passes `pl-device-test`. QS-1202 can be developed alongside once QS-502 has merged.

**Scope changes in this plan (need PO approval):**
- **QS-504 drops `backprop`.** NumPy kernels can't be traced by autodiff frameworks. Backprop comes with the JAX
  backend (QS-606), and the device gives a clear error until then (ADR-0005 D4).
- **QS-1202 narrows to the first three examples.** The QML classifier and batching examples need native
  broadcasting (QS-506, Sprint 6).

## Test plan

| ID | Verification |
|---|---|
| QS-502 | Entry point resolves (`qml.device("ketjet.qubit", wires=3)`); `import ketjet` doesn't import PennyLane (fresh subprocess); **differential**: random QNodes vs `default.qubit` for `state`, `probs`, `expval`, `var` in both precisions, with arbitrary wire labels (`["a", 3, "q"]`); seeded shots are reproducible and within 4σ; shot vectors return one result per bin; `GlobalPhase` makes `state()` match exactly |
| QS-503 | Every supported gate reaches the device untranslated (no decomposition); unsupported gates and templates (`QubitUnitary`, `StatePrep`, `StronglyEntanglingLayers`, `QFT`) decompose and match `default.qubit`; an operation with no decomposition raises a `DeviceError` naming it |
| QS-504 | Adjoint Jacobians and VJPs match `default.qubit` backprop on random QNodes, including decomposed gates (`qml.adjoint(qml.RX(x))`, `QubitUnitary`) and `CRot`, which proves the 1:1 rule; `"best"` picks adjoint; adjoint rejects `var`, `probs` and shots with a hint; `backprop` raises PennyLane's `QuantumFunctionError` (the adjoint hint is in the device docs); **counting**: one `adjoint_jacobian` call per tape for a VJP over M measurements; PennyLane's parameter-shift and finite differences work, with and without shots |
| QS-505 | `pl-device-test --device=ketjet.qubit` passes in exact and stochastic modes in a required CI job; every skip is listed with its reason in `docs/pennylane.md` |
| QS-1203 | README examples are executed by a test (as the H₂ page is) |
| QS-1205 | Dry run: the tag pipeline publishes to TestPyPI, and the smoke test installs `ketjet[pennylane]` on 3 OSes and runs a QNode; then the real release to PyPI after PO approval of the `pypi` environment; the GitHub Release has the changelog |
| QS-1202 | Each example runs in CI and asserts its result |
| Sprint | **Integration:** the H₂ VQE from Sprint 3, rewritten as a PennyLane QNode on `ketjet.qubit` with `diff_method="adjoint"` and `qml.GradientDescentOptimizer`, reaches the FCI energy |

## Demo

```bash
pip install "ketjet[pennylane]"
```

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("ketjet.qubit", wires=2)

@qml.qnode(dev, diff_method="adjoint")
def circuit(x):
    qml.RY(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

x = np.array(0.3, requires_grad=True)
circuit(x), qml.grad(circuit)(x)   # (cos 0.3, -sin 0.3)
```

## Actions needed from the PO

1. **Approve ADR-0005**, including the two scope changes above, by merging this PR. Device code starts after that.
2. **Approve the PyPI publish** when v0.1.0 is tagged. The release workflow pauses at the `pypi` environment until
   you click **Approve** in GitHub Actions. Publishing to PyPI is public and permanent (a version can't be
   re-uploaded), so it's your call, made at release time.
