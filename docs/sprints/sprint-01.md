# Sprint 01: The engine

| | |
|---|---|
| **Status** | Planned (awaiting ADR approval) |
| **Started** | 2026-11-09 |
| **Milestone** | `Sprint 01 – The engine` |
| **Sprint goal** | Apply any supported gate to any qubits of an n-qubit state, in either precision, with correctness proven against independent references. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-201 | ADR-0001: statevector representation, qubit ordering, precision | 2 | n/a | Planning PR (this) |
| QS-301 | ADR-0002: Circuit IR | 2 | n/a | Planning PR (this) |
| QS-202 | Allocate an n-qubit $\|0\dots0\rangle$ state in either precision; memory helper | 1 | QS-201 | PR A |
| QS-203 | Generic `apply_matrix(state, U, wires)` for k-qubit gates | 5 | QS-202 | PR B |
| QS-204 | Standard gate library: `Operation`, `GateSpec`, registry | 5 | QS-301, QS-203 | PR C |
| QS-209 | Typed `Backend` protocol + NumPy reference backend | 3 | QS-201, QS-204 | PR D |
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

*(Completed at sprint close.)*

## Retrospective

*(Keep / change / try.)*
