# Sprint 03: Differentiation

| | |
|---|---|
| **Status** | In review |
| **Review** | 2026-11-11 |
| **Started** | 2026-11-10 |
| **Milestone** | `Sprint 03 – Differentiation` |
| **Sprint goal** | A user can get exact gradients of any expectation value three ways (adjoint, parameter-shift, finite differences) that agree with each other and with PennyLane, with adjoint memory proven independent of circuit depth. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-403 | ADR-0004: differentiation (adjoint, shift rules, VJP) | 2 | n/a | #118 |
| QS-401 | `ketjet.gradients` front end (D1, D2) and central finite differences | 2 | ADR-0004 | #119 |
| QS-402 | Parameter-shift from generator spectra, including four-term rules for controlled rotations | 3 | QS-401 | #120 |
| QS-404 | Adjoint differentiation as a backend capability, with a memory test | 8 | QS-402 | #121 |
| QS-406 | Vector-Jacobian products, adjoint in one backward pass | 3 | QS-404 | #122 |
| QS-205 | Specialized kernels for diagonal and permutation gates, with a benchmark | 5 | n/a | #123 |
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
to the known ground-state energy. See [Sprint review: Demo](#demo_1) for the result.

## Sprint review

**Goal met? Yes.** A user can get exact gradients of any expectation value three ways (adjoint, parameter-shift,
finite differences). They agree with each other and with PennyLane's backprop, and adjoint memory is measured to be
independent of circuit depth.

| ID | Outcome |
|---|---|
| QS-403 | ✅ Done (#118): ADR-0004 approved; complex64 finite-difference tolerance approved in #119 |
| QS-401 | ✅ Done (#119): `grad`, `jacobian`, `Circuit.gradient`; chain rule over shared symbols; finite differences |
| QS-402 | ✅ Done (#120): parameter-shift from generator spectra; four-term rule for controlled rotations; shots |
| QS-404 | ✅ Done (#121): adjoint as one backend call, now the default; about 5 states of memory at any depth |
| QS-406 | ✅ Done (#122): `vjp`; adjoint in one backward pass however many observables |
| QS-205 | ✅ Done (#123): diagonal and permutation kernels, 2–10× faster from 16 qubits up |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ gradients front end, shift rules, gate derivatives, adjoint, VJP, specialized kernels |
| Property-based (Hypothesis) | ✅ declared generator spectra match the FFT of the expectation value, for every parametrized gate with and without controls; specialized kernels equal `apply_matrix` for every diagonal and permutation gate, with random controls and wires, in both precisions |
| Differential | ✅ adjoint, parameter-shift and finite differences vs `default.qubit` backprop on 15 random circuits; the H₂ Hamiltonian vs a fresh `qml.qchem` build |
| Integration | ✅ gradient descent reaches the exact minimum of a two-qubit Ising model to 1e-8; the **H₂ VQE** reaches the FCI energy to 1e-12 |
| **Regression** | ✅ new golden Jacobian (`gradient3`: shared symbols, controlled rotations, Rot and U3) cross-checked against backprop at generation; all three methods in both precisions must reproduce it |
| Memory | ✅ adjoint peak of about 5.1 to 5.5 states at depth 20 and 500 (14 qubits); about one more state per observable |
| Statistical | ✅ shot-based parameter-shift within 4σ; reproducible with a seed, with independent estimates |
| Mutation checks | ✅ 30 injected bugs across 5 PRs, all caught. One survived the first draft (QS-406, a masked finiteness check) and led to stronger tests |
| Coverage | ✅ 100% (1115 tests) |
| Docs | ✅ ADR-0004; [H₂ example](../examples/vqe-h2.md) executed by the test suite; API reference for `gradients` and `adjoint` |

**Bugs found by tests before merge:**
- **Adjoint memory:** the memory test showed 2M λ states instead of M (#121).
- **Two-term rule:** a test that only measured the target qubit would have hidden the wrong shift rule for controlled rotations. It now measures the control (#120).
- **Signed zeros:** Linux and Windows doctests printed `-0.`, twice (#121, #123). The math was right; only the printing differed by platform.

### Benchmark deltas

From #123 (Apple M2, complex128, best of 7): specialized kernels vs the dense kernel. At 20 to 22 qubits they're
4–7× faster for controlled and phase gates and 2–3× for full-state gates (RZ, X, SWAP). At 10 qubits, per-call
overhead dominates and X is 0.8×. That's one machine; the methodology ADR and asv suite (QS-1101, QS-1102) come
later.

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 23 of 23 points, 6 of 6 stories (Sprint 0: 16, Sprint 1: 18, Sprint 2: 21) |
| Sprint duration | 2026-11-10 to 2026-11-11 |
| WIP limits | Held: one story PR in review at a time (serial by design) |
| Escaped defects | 0 |
| CI failures before merge | 2, both signed-zero doctests on Linux and Windows |

### Demo

[The ground state of H₂](../examples/vqe-h2.md): 4 qubits, 15 Pauli terms, a one-parameter ansatz, and 40 steps
of gradient descent with adjoint gradients.

```python
energy = ansatz.bind(values).expval(H)
print(f"{energy:.10f}")              # -1.1372697289 (FCI: -1.1372697289)
print(f"{(hartree_fock - exact) * 1000:.2f} mHa")   # 20.59 mHa of correlation energy, all recovered
```

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Chain the oracles.** Finite differences checked parameter-shift, which checked adjoint, and PennyLane checked all three. Each method landed with an independent reference already in place | **Signed zeros broke CI twice.** macOS (Accelerate) and Linux/Windows (OpenBLAS) differ in the sign of zero. Doctests that print arrays now normalize with `+ 0.0` | Run the doctests in a Linux container locally before pushing (free, with Docker or Colima) |
| **Measure the ADR's claims.** The memory test turned "2 + M states" into a number, and caught the code breaking it | **Label only fragment-less PRs `skip-changelog`.** I added it to #122 out of habit and removed it | Run `mutmut` nightly instead of hand-written mutants (carried over from Sprints 1 and 2) |
| **Run the docs.** The H₂ page is executed by the tests, and every printed line must appear on the page | Regenerating golden files rewrote an unrelated one with 1-ulp noise. It was restored; regeneration should write only the files that changed meaningfully | |

## Forecast update

Throughput over four sprints: 16, 18, 21 and 23 points (mean 19.5). 243 points remain across Sprints 4–16, which is
about 12.5 sprints at the current pace, inside the 13 sprints planned. Sprint 4 brings the PennyLane device
(QS-501 to QS-505), which rests on this sprint's `jacobian` and `vjp`, and the first PyPI release, **v0.1.0**.
