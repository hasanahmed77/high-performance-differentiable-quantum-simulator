# Sprint 02: Circuits and measurements

| | |
|---|---|
| **Status** | Done: accepted by the PO in #117 |
| **Started** | 2026-11-10 |
| **Milestone** | `Sprint 02 – Circuits and measurements` |
| **Sprint goal** | A user can build a circuit with a fluent API, bind parameters, and read every standard result (state, probabilities, expectation values, variances, samples), all matching PennyLane. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-108 | Changelog fragments (towncrier) enforced in CI | 2 | n/a | #111 |
| QS-302 | Fluent `Circuit` builder: repr, equality, serialization round-trip | 3 | n/a (ADR-0002 D6) | #112 |
| QS-303 | Symbolic parameters, binding, trainable-parameter tracking | 3 | QS-302 | #113 |
| QS-305 | Observables: Pauli strings, Hermitian, tensor products, Hamiltonians | 5 | ADR-0003 | #114 |
| QS-304 | `state()`, `probs(wires)`, `expval(obs)`, `var(obs)`; backend protocol extended | 5 | QS-302, QS-305 | #115 |
| QS-306 | `sample(shots, seed)` and shot-based expectation values | 3 | QS-304 | #116 |
| | **Total** | **21** | | |

**Sequencing:** ADR-0003 (this planning PR) must be approved before PR D. QS-108 goes first, so every later PR
ships its own changelog fragment. PRs are serial, with each one built locally while the previous PR's CI runs.

## Test plan

| ID | Verification |
|---|---|
| QS-108 | CI fails on a PR without a fragment and passes with one or with the `skip-changelog` label; `towncrier build --draft` renders the existing entries |
| QS-302 | Unit: every fluent method appends the right `Operation`; validation against `n_wires`; `operations` is an immutable snapshot; `repr`, equality and JSON round-trip (property-based over random circuits) |
| QS-303 | Unit: `parameters` order and uniqueness; `bind` by `Parameter` or name returns a new circuit and leaves the original unchanged; missing or unknown names raise; trainable indices point at the right `(operation, parameter)` slots |
| QS-305 | Unit: algebra (`@`, `*`, `+`, `-`), Pauli simplification, Hermitian validation, disjoint-wire checks; **property:** the Pauli fast path equals the dense Kronecker-product matrix on random words and states |
| QS-304 | **Differential** vs `default.qubit` on random circuits: `state`, `probs` (random wire subsets and orders), `expval` and `var` for every observable type, both precisions; **regression golden files** for QFT and GHZ outputs |
| QS-306 | Same seed gives identical samples; output shape and dtype; **chi-square** goodness of fit against `probs`; shot-based `expval` within 4σ of exact for every observable type |

## Demo

```python
import ketjet as kj
from ketjet.observables import X, Z

c = kj.Circuit(2).h(0).cnot(0, 1)
c.probs()                    # [0.5, 0, 0, 0.5]
c.expval(Z(0) @ Z(1))        # 1.0
c.sample(5, seed=7)          # five correlated bit pairs
```

## Sprint review

**Goal met? Yes.** A user can build a circuit fluently, bind parameters, and read every standard result
(state, probabilities, expectation values, variances, samples, and shot-based estimates), all matching PennyLane.

| ID | Outcome |
|---|---|
| QS-108 | ✅ Done (#111): towncrier fragments enforced in CI |
| QS-302 | ✅ Done (#112): fluent `Circuit` builder, versioned JSON |
| QS-303 | ✅ Done (#113): symbolic parameters, binding, trainable slots |
| QS-305 | ✅ Done (#114): observables with algebra, matrix-free Pauli path |
| QS-304 | ✅ Done (#115): `state`, `probs`, `expval`, `var`; backend measurement protocol; quickstart |
| QS-306 | ✅ Done (#116): seeded sampling and shot-based expectation values |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ circuit, parameters, observables, measurements, sampling |
| Property-based (Hypothesis) | ✅ JSON round-trip (200 random circuits), Pauli fast path vs `np.kron` reference (200 cases), marginal consistency |
| Differential | ✅ observable matrices vs `qml.matrix`; `probs` on random wire subsets and `expval`/`var` of random Hamiltonians vs `default.qubit` (20 seeds × 2 precisions), at standard tolerances |
| Integration | ✅ random circuits vs `default.qubit` (unchanged from Sprint 1, now on shared helpers) |
| **Regression** | ✅ golden files for QFT (4, 5 qubits) and GHZ (5), cross-checked against `default.qubit` at generation; analytic QFT formula; complex64 uniform-superposition sampling |
| Statistical | ✅ chi-square goodness of fit (α = 10⁻⁶); shot estimates within 4σ |
| Mutation checks | ✅ 7 injected bugs (Y phase, LSB wire order, dropped Sum coefficients, sample bit order, Y rotation, Hermitian rotation, no renormalization), all caught. One survived the first draft and led to the regression test |
| Coverage | ✅ 100% (707 tests) |
| Docs | ✅ quickstart with executed examples; API reference for every new module |

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 21 of 21 points, 6 of 6 stories (Sprint 0: 16, Sprint 1: 18) |
| Sprint duration | 2026-11-10 (same day) |
| WIP limits | Held: at most 2 PRs in review |
| Escaped defects | 0 |

### Demo

```python
import ketjet as kj
from ketjet.observables import X, Z

bell = kj.Circuit(2).h(0).cnot(0, 1)
bell.probs()                              # [0.5, 0, 0, 0.5]
bell.expval(Z(0) @ Z(1))                  # 1.0
bell.sample(5, seed=7)                    # [[1,1],[1,1],[1,1],[0,0],[0,0]]
bell.expval(X(0), shots=10_000, seed=7)   # -0.0034 (exact: 0)
```

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Make surviving mutants into tests.** The "no renormalization" mutant survived, and finding an input that really exercises the path uncovered a real crash (complex64 uniform superpositions are rejected by NumPy's sampler) | **Don't loosen tolerances, even in a draft.** A first draft used 10× tolerances; the standard ones were enough | Run `mutmut` nightly instead of hand-written mutants (carried over from Sprint 1) |
| **Golden files that cross-check themselves** against an independent simulator before writing | **`gh pr create`/`edit` hang or fail on this repo** (Projects-classic API); create and edit PRs through the REST API | Batch board updates in one GraphQL request (done: `board.py`), and keep retrying flaky network calls instead of waiting |
| Executing every docs snippet before committing: it caught a quickstart example that printed the same value for every parameter | Order merges code-first, docs-second, so the cheap docs CI is the one that re-runs after a rebase | |

## Forecast update

Throughput over three sprints: 16, 18 and 21 points (mean 18.3). 266 points remain across Sprints 3–16, which is
about 14.5 sprints at the current pace, consistent with the 14 sprints planned. Sprint 3 (differentiation, 23 points)
is the first sprint above the mean; split stories if review wait grows.
