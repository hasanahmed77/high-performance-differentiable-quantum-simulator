# Sprint 04: PennyLane device and v0.1.0

| | |
|---|---|
| **Status** | In review |
| **Review** | 2026-11-12 |
| **Started** | 2026-11-11 |
| **Milestone** | `Sprint 04 – PennyLane device` |
| **Sprint goal** | Anyone can `pip install "ketjet[pennylane]"` from PyPI and run, and differentiate, any PennyLane QNode on `qml.device("ketjet.qubit")`, with PennyLane's own device test suite passing. Release **v0.1.0**. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-501 | Spike and ADR-0005: the PennyLane device design | 2 | n/a | #126 |
| QS-502 | `qml.device("ketjet.qubit")` via an entry point; tapes translate 1:1 and execute; analytic and shot measurements | 5 | ADR-0005 | #127 |
| QS-503 | Unsupported operations decompose automatically; clear errors for anything that can't | 3 | QS-502 | #128 |
| QS-504 | Device gradients: `diff_method="adjoint"` and single-pass VJP (backprop deferred to the JAX backend, ADR-0005 D4) | 5 | QS-503 | #129 |
| QS-505 | `pl-device-test` passing in CI, with skips documented | 3 | QS-504 | #130 |
| QS-1203 | README for the release: problem, design, results, install, quickstart, badges | 2 | QS-505 | #131 |
| QS-1205 | Release pipeline dry run (TestPyPI and smoke tests) and the real **v0.1.0** release | 5 | QS-1203 | #132, tag `v0.1.0` |
| QS-1202 | First runnable examples: Bell, QFT and VQE scripts on `ketjet.qubit`, executed in CI (**stretch**) | 3 | QS-502 | #133 |
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

## Sprint review

**Goal met? Yes.** `pip install "ketjet[pennylane]"` installs [ketjet 0.1.0 from PyPI](https://pypi.org/project/ketjet/0.1.0/),
and any PennyLane QNode runs and differentiates on `qml.device("ketjet.qubit")`. PennyLane's own device test suite
passes with no failures, on every pull request.

| ID | Outcome |
|---|---|
| QS-501 | ✅ Done (#126): ADR-0005 approved; amended twice with PO approval (`CRot` runs directly, #127; backprop gets PennyLane's own error, #129) |
| QS-502 | ✅ Done (#127): `ketjet.qubit` via entry point, behind the `[pennylane]` extra; 1:1 translation; native `expval`/`var`/`probs`; seeded shots |
| QS-503 | ✅ Done (#128): 22 unsupported operations and templates decompose and match `default.qubit`; clear errors otherwise |
| QS-504 | ✅ Done (#129): device adjoint and one-pass VJP; `"best"` picks adjoint; backprop deferred to the JAX backend (ADR-0005 D4) |
| QS-505 | ✅ Done (#130): `pl-device-test` passes (380 exact, 365 with shots, 0 failed) as a required CI job; every skip documented |
| QS-1203 | ✅ Done (#131): release README, its examples executed by a test, links valid on PyPI |
| QS-1205 | ✅ Done (#132, tag `v0.1.0`): TestPyPI dry run, smoke tests on 3 OSes, **v0.1.0 on PyPI** after PO approval, GitHub Release |
| QS-1202 | ✅ Done, stretch (#133): Bell, QFT and VQE examples, each checking its own result in CI |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ translation, observables, supported-set rules, derivative formats, VJP batching |
| Differential | ✅ random QNodes vs `default.qubit` (state, probs, expval, var; 15 seeds × 2 precisions; arbitrary wire labels); adjoint Jacobians and VJPs vs backprop (10 seeds × 2 precisions, shared parameters, decomposed gates) |
| **Conformance** | ✅ PennyLane's `pl-device-test`: exact 380 passed / 37 skipped / 0 failed; 10,000 shots 365 / 52 / 0. Required CI job |
| Integration | ✅ the sprint's integration test: an H₂ VQE as a QNode with adjoint gradients reaches the FCI energy (`examples/vqe.py`, error 1.6e-15); README and example scripts executed by tests |
| Regression | ✅ every conformance-suite bug pinned in the main matrix (`test_pennylane_conformance.py`) |
| Packaging | ✅ wheel installs and imports; entry point registered; the full suite passes from the unpacked sdist (1262 tests); `twine check` |
| Release | ✅ TestPyPI smoke tests on Linux, macOS and Windows ran a QNode with adjoint gradients; a clean install from PyPI reproduced the README quickstart exactly; the PyPI files' SHA-256 equal the reviewed ones |
| Mutation checks | ✅ 39 injected bugs across 6 PRs, all caught. Three survived a first draft (two in #129, one in #130), and each led to a stronger test |
| Coverage | ✅ 100% (1266 tests) |
| Docs | ✅ ADR-0005; PennyLane device page with conformance results; example scripts page; release README |

**Bugs found before release** (none reached PyPI):
- **Multi-wire `qml.Identity`** crashed the device: support was matched by name only (#128).
- **A trainable observable parameter** in a direct device call raised `IndexError` (#129). Two tests had been passing
  for the wrong reason; mutation checks exposed both.
- **PennyLane's device suite found three bugs my own tests missed** (#130). All three had already merged in #127:
  - non-diagonal observables outside ketjet's native path weren't rotated into their eigenbasis, so `hadamard`
    gradients of probabilities were **silently zero**;
  - `Hermitian` and `Projector` were sampled in the wrong basis;
  - operations with a matrix but no decomposition failed.
- **A flaky Sprint 3 memory test** failed whenever run on its own. It measured a first call with a growing parameter
  count. Measured in steady state, memory is 5.1–5.35 states from depth 20 to 4000 (#130).
- **Changelog contradictions:** two fragments each called a different method "the default". They were reworded before
  the release (#132).

### Release

| | |
|---|---|
| Version | **0.1.0** (Alpha), [PyPI](https://pypi.org/project/ketjet/0.1.0/), [GitHub Release](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/releases/tag/v0.1.0) |
| Files | `ketjet-0.1.0-py3-none-any.whl` (55.7 KB, SHA-256 `9bdb3cfc…`), `ketjet-0.1.0.tar.gz` (101.8 KB, `bf409457…`) |
| Approval | PO reviewed the files and approved the `pypi` environment |
| Quality bar | QB1 (README) and QB2 (PyPI and changelog) first met; QB6 (docs) extended |

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 28 of 28 points (25 committed and the 3-point stretch), 8 of 8 stories (Sprint 0: 16, Sprint 1: 18, Sprint 2: 21, Sprint 3: 23) |
| Sprint duration | 2026-11-11 to 2026-11-12 |
| WIP limits | Held: one story PR in review at a time (serial by design) |
| Escaped defects | 0 released. Three defects reached `main` (#127) and were caught before the release by the conformance suite (#130) |
| CI failures before merge | 1: Python 3.14+ turned a dependency's (autograd's) compile-time warning into a collection error (#127) |
| ADR amendments | 2, both approved by the PO before merging |

### Demo

```bash
pip install "ketjet[pennylane]"
```

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("ketjet.qubit", wires=2)

@qml.qnode(dev, diff_method="adjoint")
def energy(theta):
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(theta[1], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1))

theta = np.array([0.3, 0.8], requires_grad=True)
print(f"{energy(theta):.6f}")             # 0.844467  (installed from PyPI)
print(qml.grad(energy)(theta).round(6))   # [ 0.477668 -0.717356]
```

And `python examples/vqe.py` finds the H₂ ground state, −1.13726973 Ha, to 1.6e-15.

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Run the official conformance suite.** `pl-device-test` found three real bugs that our own 1,200+ tests missed, one of them a silently wrong gradient | **Run conformance first, not last.** The suite arrived in QS-505, after its bugs had merged in QS-502. The next backend runs it from its first PR | Add `pl-device-test` with JAX installed once PennyLane supports JAX 0.12 (Sprint 5) |
| **Review release artifacts by hash.** The PO approved bytes already tested on TestPyPI and 3 OSes, then confirmed the same bytes on PyPI | **Use generic, asymmetric test states.** A symmetric state hid the eigenbasis bug, and a test whose last operation was a global phase hid a mapping bug | Run `mutmut` nightly instead of hand-written mutants (carried over since Sprint 1) |
| **Flag ADR deviations to the PO before merging.** Two amendments, both approved, neither surprising | **Isolate measurement tests from test order.** The memory test depended on what ran before it. Steady-state measurement fixed it | Run the doctests in a Linux container before pushing (carried over) |

## Forecast update

Throughput over five sprints: 16, 18, 21, 23 and 28 points (mean 21.2). 215 points remain across Sprints 5–16, which is
about 10.1 sprints at the current pace, against the 12 planned. Two sprints of slack is welcome before the native and
CUDA work (Sprints 8–13), where the risk is highest. Sprint 5 starts the JAX backend, with `pl-device-test` from its
first PR.
