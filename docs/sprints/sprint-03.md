# Sprint 03: Differentiation

| | |
|---|---|
| **Status** | In progress |
| **Started** | 2026-11-10 |
| **Milestone** | `Sprint 03 – Differentiation` |
| **Sprint goal** | A user can get exact gradients of any expectation value three ways (adjoint, parameter-shift, finite differences) that agree with each other and with PennyLane, with adjoint memory proven independent of circuit depth. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-403 | ADR-0004: differentiation (adjoint, shift rules, VJP) | 2 | n/a | this PR |
| QS-401 | `ketjet.gradients` front end (D1, D2) and central finite differences | 2 | ADR-0004 | |
| QS-402 | Parameter-shift from generator spectra, including four-term rules for controlled rotations | 3 | QS-401 | |
| QS-404 | Adjoint differentiation as a backend capability, with a memory test | 8 | QS-402 | |
| QS-406 | Vector-Jacobian products, adjoint in one backward pass | 3 | QS-404 | |
| QS-205 | Specialized kernels for diagonal and permutation gates, with a benchmark | 5 | n/a | |
| | **Total** | **23** | | |

**Sequencing:** ADR-0004 (this planning PR) must be approved before any gradient code. Gradient PRs are serial,
because each one's tests use the previous method as an oracle: finite differences check parameter-shift, and
parameter-shift checks adjoint. QS-205 is independent and lands last, so the full gradient suite guards the new
kernels.

## Test plan

| ID | Verification |
|---|---|
| QS-401 | Unit: shapes `(P,)` and `(M, P)`, parameter order, shared symbols sum their slots, missing or unknown values raise, unsupported method/measurement pairs raise with a hint; finite differences equal analytic derivatives of single-gate circuits ($\partial_\theta \cos\theta$, etc.) |
| QS-402 | **Spectrum property test:** the Fourier transform of $f(x)$ has only the declared frequencies, for every parametrized gate with and without controls; parameter-shift equals finite differences on random circuits; the two-term rule is shown to be **wrong** for a controlled rotation (guards against regressing to it); shots: reproducible with a seed and within 4σ of exact |
| QS-404 | `derivative_fn` equals central differences of `matrix_fn`; adjoint equals parameter-shift on random circuits (controls, shared symbols, every observable type, both precisions, depth up to 500); **differential** vs `default.qubit` with `diff_method="backprop"`; **memory:** peak traced allocation bounded by a small multiple of the state size and equal at depth 20 and 500 |
| QS-406 | VJP equals `jacobian(...).T @ v` for random cotangents, for every method; the adjoint VJP makes one backward pass regardless of the number of observables |
| QS-205 | Property: specialized kernels equal `apply_matrix` for every diagonal and permutation gate, with random controls and wires, both precisions; the full suite (including gradients) passes unchanged; benchmark of speedup vs the dense kernel recorded in the PR |
| Sprint | **Integration:** a gradient-descent loop on a parametrized circuit converges to the known minimum; regression golden file for one gradient |

## Demo

A VQE-style optimization on the H₂ molecule: gradient descent with adjoint gradients on a Hamiltonian, converging
to the known ground-state energy.

```python
import ketjet as kj
from ketjet import gradients
from ketjet.observables import Z
from ketjet.operations import Parameter

theta = Parameter("theta")
c = kj.Circuit(1).ry(theta, 0)
gradients.grad(c, Z(0), {"theta": 0.3})  # [-sin(0.3)]
```
