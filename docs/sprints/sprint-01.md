# Sprint 01: The engine

| | |
|---|---|
| **Status** | Done: accepted by the PO in #109 |
| **Started** | 2026-11-09 |
| **Milestone** | `Sprint 01 – The engine` |
| **Sprint goal** | Apply any supported gate to any qubits of an n-qubit state, in either precision, with correctness proven against independent references. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-201 | ADR-0001: statevector representation, qubit ordering, precision | 2 | n/a | #104 |
| QS-301 | ADR-0002: Circuit IR | 2 | n/a | #104 |
| QS-202 | Allocate an n-qubit $\|0\dots0\rangle$ state in either precision; memory helper | 1 | QS-201 | #105 |
| QS-203 | Generic `apply_matrix(state, U, wires)` for k-qubit gates | 5 | QS-202 | #106 |
| QS-204 | Standard gate library: `Operation`, `GateSpec`, registry | 5 | QS-301, QS-203 | #107 |
| QS-209 | Typed `Backend` protocol + NumPy reference backend | 3 | QS-201, QS-204 | #108 |
| | **Total** | **18** | | |

**Sequencing:** the planning PR (ADRs) must be approved before PR A starts (Definition of Ready, plan.md §5.1).
PRs A → D are serial because each builds on the previous one, which also keeps In Review within its WIP limit.
QS-105 was closed by #103 before this sprint started.

## Test plan

| ID | Verification |
|---|---|
| QS-202 | Unit: shape, dtype and $\|0\dots0\rangle$ amplitude for both precisions; `state_nbytes` exact; invalid `n` and `max_bytes` overflow raise clear errors |
| QS-203 | Unit: known results (X on each wire of a 3-qubit state, H⊗H); **property (Hypothesis):** norm preserved, U then U† gives the identity, any wire order equals the explicit Kronecker-product construction; dtype preserved (complex64 stays complex64) |
| QS-204 | Unit: every gate's matrix is unitary for random parameters; **differential:** matrices match PennyLane's `qml.matrix` (wire order included); structure flags (`is_diagonal`, `is_permutation`, `is_self_inverse`) verified against the matrix; one test per IR validation rule |
| QS-209 | Unit: the NumPy backend satisfies the protocol (checked by mypy and at runtime); **integration:** Bell and GHZ states produced end to end through the backend; consumed-input rule documented and the reference backend proven pure (input unchanged) |

Shared tolerances (`tests/tolerances.py`, plan.md §6.3) are introduced in PR A.

## Demo

Bell $(|00\rangle + |11\rangle)/\sqrt{2}$ and 3-qubit GHZ states built through the backend, with amplitudes matching
PennyLane's `default.qubit`.

## Risks for this sprint

- **PennyLane on Python 3.15:** PennyLane doesn't list 3.15 support yet. Differential tests are marked `pennylane`
  and skip cleanly when PennyLane can't be installed (D4, plan.md §12).

## Sprint review

**Goal met? Yes.** Any of the 20 base gates, plus any controlled form, can be applied to any wires of an n-qubit
state in either precision. Correctness is proven against three independent references: a bit-arithmetic dense
operator, PennyLane's gate matrices, and PennyLane's `default.qubit` simulator.

| ID | Outcome |
|---|---|
| QS-201 | ✅ Done (#104): ADR-0001 accepted |
| QS-301 | ✅ Done (#104): ADR-0002 accepted |
| QS-202 | ✅ Done (#105): `zero_state`, `state_nbytes`, clear memory errors, shared tolerances |
| QS-203 | ✅ Done (#106): reference `apply_matrix` kernel |
| QS-204 | ✅ Done (#107): 20 base gates, controlled gates as structure, `Operation`, `Parameter` |
| QS-209 | ✅ Done (#108): `Backend` protocol and `NumPyBackend` |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ statevector, kernel, gates, operations, backend |
| Property-based (Hypothesis) | ✅ kernel vs independent dense reference (150 cases), norm preservation, inverse round-trips, dtype preservation |
| Differential | ✅ 32 gate cases × 2 precisions equal PennyLane's `qml.matrix`, wire order included |
| Integration | ✅ Bell, GHZ (3/5/8 qubits) and 25 random 30-gate circuits × 2 precisions equal `default.qubit` |
| Regression | n/a: no escaped bugs in simulator code yet |
| Mutation checks | ✅ 7 injected bugs (axis move, transposed matrix, dtype promotion, false `is_diagonal`, RY sign, control block, ignored controls), all caught |
| Coverage | ✅ 100% (395 tests) |
| Docs build | ✅ strict; ADRs and sprint reports published; math rendered with KaTeX |
| `main` CI after every merge | ✅ |

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 18 of 18 points, 6 of 6 stories |
| Median PR open → merge | 6 min for code PRs after the merge policy change (the ADR PR took 41 min, including PO review) |
| Sprint duration | 2026-11-10 (same day) |
| WIP limits | Held: at most 1 PR in review at a time (serial PRs, built locally while CI ran) |
| Escaped defects | 1 from Sprint 0: math on the docs site never rendered (MathJax was never loaded). Fixed in #104 |

### Demo

```python
from ketjet.backends import NumPyBackend
from ketjet.operations import CNOT, H

backend = NumPyBackend()
bell = backend.apply_operations(backend.zero_state(2), [H(0), CNOT(0, 1)])
backend.to_numpy(bell).round(3)   # [0.707, 0, 0, 0.707], identical to PennyLane default.qubit
```

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Mutation checks** on every core module: they proved that the independent references (not the property tests) catch subtle bugs such as a transposed matrix | **Check the reference side of differential tests as carefully as the code:** the PennyLane harness double-queued every gate, and all 55 integration tests failed before the simulator was even at fault | Add a lightweight mutation-testing tool (e.g. `mutmut`) to the nightly workflow instead of manual mutants |
| **Building the next PR locally while CI runs**, then rebasing with `--onto` | **Watch formatter scope:** ruff 0.17 rewrites code inside Markdown and tried to edit an accepted ADR; Markdown is now excluded | Write a short "Sprint 2 risks" note before planning (Circuit API ergonomics, towncrier setup) |
| ADRs before code: no Sprint 1 story needed a design change after implementation began | GitHub only closes the *first* issue in "Closes #a and #b"; use one `Closes #n` per line | |

## Forecast update

Throughput: Sprint 0 delivered 16 points and Sprint 1 delivered 18. The first throughput-based forecast follows
Sprint 2, as planned.
