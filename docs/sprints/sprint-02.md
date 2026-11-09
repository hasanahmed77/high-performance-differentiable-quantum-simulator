# Sprint 02: Circuits and measurements

| | |
|---|---|
| **Status** | Planned (awaiting ADR-0003 approval) |
| **Started** | 2026-11-10 |
| **Milestone** | `Sprint 02 – Circuits and measurements` |
| **Sprint goal** | A user can build a circuit with a fluent API, bind parameters, and read every standard result (state, probabilities, expectation values, variances, samples), all matching PennyLane. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-108 | Changelog fragments (towncrier) enforced in CI | 2 | n/a | PR A |
| QS-302 | Fluent `Circuit` builder: repr, equality, serialization round-trip | 3 | n/a (ADR-0002 D6) | PR B |
| QS-303 | Symbolic parameters, binding, trainable-parameter tracking | 3 | QS-302 | PR C |
| QS-305 | Observables: Pauli strings, Hermitian, tensor products, Hamiltonians | 5 | ADR-0003 | PR D |
| QS-304 | `state()`, `probs(wires)`, `expval(obs)`, `var(obs)`; backend protocol extended | 5 | QS-302, QS-305 | PR E |
| QS-306 | `sample(shots, seed)` and shot-based expectation values | 3 | QS-304 | PR F |
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

*(Completed at sprint close.)*

## Retrospective

*(Keep / change / try.)*
