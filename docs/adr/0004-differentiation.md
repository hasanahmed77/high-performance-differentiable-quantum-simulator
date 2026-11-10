# ADR-0004: Differentiation

| | |
|---|---|
| **Status** | Proposed (awaiting PO approval) |
| **Date** | 2026-11-10 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-401, QS-402, QS-403, QS-404, QS-406; later QS-405, QS-407, QS-504 (PennyLane), QS-606 (JAX), QS-707 (native), QS-805 (fusion), QS-904 (GPU) |

## Context

Gradients are why ketjet exists (plan.md G2): variational algorithms (VQE, QAOA, QML) spend nearly all their time
evaluating $\partial \langle O \rangle / \partial \theta$. Sprint 3 delivers three methods, and every later backend
(JAX, native, CUDA) and the PennyLane device build on the choices made here. The forces:

- **Exactness:** analytic gradients must agree with each other to working precision, and with finite differences
  to the plan's gradient tolerance ($10^{-5}$, plan.md §6.3).
- **Memory:** deep circuits must differentiate in $O(1)$ statevector copies, independent of depth (G2). Storing
  every intermediate state, as reverse-mode autodiff does, costs $O(\text{depth} \cdot 2^n)$ and caps us around 20
  qubits on a laptop.
- **Hardware compatibility:** parameter-shift must give exact gradients from circuit evaluations alone, including
  controlled rotations, whose shift rule differs from that of plain rotations.
- **Composability:** the PennyLane device (QS-504) and JAX (QS-606) consume vector-Jacobian products, not full
  Jacobians.
- **Future backends:** native and GPU backends must run the adjoint pass on their own memory, with no host
  round-trips per gate.

## Decisions

### D1. Gradients are taken with respect to symbols, computed per slot

A circuit's trainable inputs are its `Parameter` symbols, in `circuit.parameters` order (ADR-0002). Each method
first computes the Jacobian with respect to every **slot** (`circuit.parameter_slots`: one entry per gate
parameter that holds a symbol), then sums slots that share a symbol, which is the chain rule:

$$\frac{\partial f}{\partial \theta} = \sum_{s \,:\, \text{symbol}(s) = \theta} \frac{\partial f}{\partial x_s}.$$

Slot-level Jacobians are what the PennyLane device needs (it tracks trainable gate parameters, not symbols), so
they are a first-class internal result, not an implementation detail.

### D2. One front end, three methods

`ketjet.gradients` exposes:

```python
grad(circuit, observable, values, *, method="adjoint")            # shape (P,)
jacobian(circuit, observables, values, *, method="adjoint")       # shape (M, P)
vjp(circuit, observables, values, cotangent, *, method="adjoint") # shape (P,), equals J.T @ cotangent
```

plus the convenience method `Circuit.gradient(observable, values, *, method=...)`. `values` maps every symbol
(or its name) to a float, exactly like `Circuit.bind`; missing symbols raise. Results are `float64` arrays, like
`expval`'s Python floats. `P` is `len(circuit.parameters)`; a circuit without symbols has `P = 0`.

| Method | Exact? | Cost per gradient | Supports | Use |
|---|---|---|---|---|
| `"adjoint"` | Yes | ~2 circuit executions, 3 states | `expval` of any observable | Default on simulators |
| `"parameter-shift"` | Yes | 2 executions per slot (4 for controlled rotations) | `expval`, `probs`; with `shots` | Hardware-compatible reference |
| `"finite-diff"` | No ($O(h^2)$) | 2 executions per slot | Any real-valued output | Test oracle and fallback only |

An unsupported combination (for example `var` with adjoint) raises a `ValueError` that names a method which does
support it. There is no silent fallback.

### D3. Adjoint differentiation (Jones and Gacon, 2020)

For $f = \langle \psi | O | \psi \rangle$ with $|\psi\rangle = U_N \cdots U_1 |0\rangle$, and $|\psi_k\rangle$ the
state after gate $k$:

$$\frac{\partial f}{\partial x_s} = 2\,\operatorname{Re}\,\big\langle \lambda_k \big|\, \partial_{s} U_k \,\big|\, \psi_{k-1} \big\rangle,
\qquad \langle \lambda_k | = \langle \psi | O \, U_N \cdots U_{k+1}.$$

Both $|\psi_{k-1}\rangle$ and $|\lambda_k\rangle$ are obtained by *undoing* gates one at a time instead of storing
them:

```text
ψ ← U_N ⋯ U_1 |0⟩              forward pass; intermediates are not kept
λ ← O ψ                         ADR-0003 D2 primitive
for k = N … 1:
    ψ ← U_k† ψ                  ψ is now |ψ_{k-1}⟩
    for each trainable slot s of gate k:
        μ ← (∂_s U_k) ψ
        g[s] ← 2 Re ⟨λ | μ⟩
    λ ← U_k† λ
```

- **Memory:** three statevectors ($\psi$, $\lambda$, $\mu$) plus kernel temporaries, **independent of depth**.
  With $M$ observables it is $2 + M$ (QS-405). Time is about two circuit executions plus one derivative
  application per slot.
- **Derivatives:** every parametrized `GateSpec` gains a `derivative_fn` giving the analytic
  $\partial U / \partial x_j$ for each parameter. For a controlled operation the derivative is
  $\partial U$ on the control-active block and **zero** elsewhere (not the identity), built by
  `Operation.derivative_matrix(j)`. The kernel `apply_matrix` already accepts non-unitary matrices.
- **Rounding:** undoing gates accumulates rounding error linearly in depth. Validation (below) checks agreement
  with parameter-shift at the standard tolerances up to depth 500.

| Option | Pros | Cons |
|---|---|---|
| **A. Adjoint with uncomputation** ✅ | $O(1)$ states in depth; ~2× forward cost; the method `lightning.qubit` uses | Expectation values only; rounding drift grows with depth |
| B. Reverse-mode autodiff (store every state) | Any differentiable output | $O(\text{depth} \cdot 2^n)$ memory; available later via JAX (QS-604) anyway |
| C. Parameter-shift only | Simple; hardware-compatible | $2P$ executions: about 100× the adjoint's cost for 100 parameters |

### D4. Adjoint is a backend capability

`Backend` (ADR-0001 D4) gains one method:

```python
adjoint_jacobian(n_wires, operations, observables, trainable, *, dtype) -> ndarray  # (M, len(trainable))
```

`operations` are bound and `trainable` lists `(operation index, parameter index)` slots. `NumPyBackend`
implements it with the reference algorithm in `ketjet.gradients`; native and CUDA backends will implement it
on their own memory (QS-707, QS-904), and JAX may implement it through `jax.custom_vjp` (QS-606).

| Option | Pros | Cons |
|---|---|---|
| **A. One coarse protocol method** ✅ | GPU and native backends fuse the whole backward pass with no per-gate host calls; one gradient test suite runs on every backend | Each backend implements the loop |
| B. Generic loop over new fine-grained primitives (`inner`, `apply_derivative`, `copy`) | One implementation | Per-gate Python dispatch dominates on GPU; widens the protocol by more methods |

Parameter-shift and finite differences need no protocol change: they only run circuits.

### D5. Parameter-shift rules come from the generator spectrum

A gate parameter enters as $U(x) = A\, e^{-i x G} B$. If the distinct differences of $G$'s eigenvalues are the
equidistant frequencies $\Omega, 2\Omega, \ldots, R\Omega$, then $f(x)$ is a trigonometric polynomial with those
frequencies, and its exact derivative is (Wierichs et al., 2022):

$$f'(x) = \sum_{\mu=1}^{R} c_\mu \big[f(x + s_\mu) - f(x - s_\mu)\big], \quad
s_\mu = \frac{(2\mu - 1)\pi}{2R\Omega}, \quad c_\mu = \frac{\Omega\,(-1)^{\mu-1}}{4R\sin^2\!\big(\tfrac{(2\mu-1)\pi}{4R}\big)}.$$

- Each parametrized `GateSpec` declares its generator eigenvalues per parameter (for example $\{\pm\tfrac12\}$
  for RX, $\{0, 1\}$ for PhaseShift). This gives $R = 1$, $s = \pi/2$, $c = \tfrac12$: the familiar two-term rule.
- **Controls add the eigenvalue 0.** A controlled rotation's generator has eigenvalues $\{-\tfrac12, 0, \tfrac12\}$,
  so its frequencies are $\{\tfrac12, 1\}$ and $R = 2$. That yields the four-term rule with shifts $\pm\pi/2$ and
  $\pm 3\pi/2$. A controlled PhaseShift stays two-term. Nothing is special-cased per gate, and new gates get correct
  rules by declaring their spectrum.
- A spectrum whose frequencies are not equidistant is rejected when the registry is built, so a wrong rule can't
  be produced silently.
- Shifts apply to **one slot at a time** (D1), so a symbol shared by several gates is handled correctly.
- With `shots`, each shifted evaluation uses an independent child seed spawned from the caller's seed
  (`numpy.random.SeedSequence.spawn`), so results are reproducible and the evaluations are uncorrelated.

| Option | Pros | Cons |
|---|---|---|
| **A. Rules derived from declared spectra** ✅ | One general formula; controlled gates correct by construction; testable property | Spectra must be declared and verified |
| B. Hand-written rule per gate | Explicit | Easy to get controlled gates wrong (the two-term rule is silently biased for them) |
| C. Decompose controlled rotations first | Two-term rules only | Changes the circuit; more executions |

### D6. VJPs use one backward pass

For cotangent $v$ over $M$ observables, $v^\top J$ is the gradient of $\sum_m v_m \langle O_m \rangle$, which is
the expectation value of the single observable $\sum_m v_m O_m$ (a `Sum`, ADR-0003 D1). The adjoint VJP is
therefore **one** backward pass with three states, whatever $M$ is. Parameter-shift and finite differences compute
`jacobian` and contract it.

### D7. Finite differences are a test oracle

Central differences, $f'(x) \approx [f(x+h) - f(x-h)] / 2h$, with $h = 10^{-7}$ for complex128 (plan.md §6.3) and
$h = 10^{-3}$ for complex64, where the rounding error of a smaller step would dominate. The step is overridable.
Finite differences are documented as approximate and are never the default.

## Consequences

- **Positive:** depth-independent gradient memory from the first release. One backend method covers native and
  GPU adjoint later. Exact parameter-shift rules for controlled gates fall out of one formula. VJP cost doesn't
  grow with the number of observables, which matters for QML losses over many outputs.
- **Negative:** each backend implements the adjoint loop. Every parametrized gate needs a derivative and a
  spectrum, both verified by tests. Adjoint supports expectation values only; variance and probability gradients
  need parameter-shift or finite differences.
- **Follow-ups:** QS-405 (multi-observable single pass), QS-407 (VQE), QS-504 (PennyLane `compute_derivatives` /
  `compute_vjp` reuse D1 slots and D6), QS-606, QS-707, QS-904; variance gradients via parameter-shift.

## Validation

- **Derivatives:** each `derivative_fn` equals central differences of `matrix_fn`, for random parameters.
- **Spectra:** for random states, observables and parameters, sampling $f(x)$ on a grid and taking its discrete
  Fourier transform shows no frequency outside the declared set, for plain and controlled gates.
- **Agreement:** on random circuits over every parametrized gate, with controls and shared symbols, adjoint
  equals parameter-shift at the standard tolerances (both precisions), and both match finite differences within
  $10^{-5}$ (complex128). Every observable type is covered, Hamiltonians included.
- **Independent oracle:** gradients equal PennyLane's `default.qubit` with `diff_method="backprop"`.
- **Memory:** the adjoint pass's peak traced allocation is at most a fixed small multiple of one state's size,
  and the same at depth 20 and depth 500.
- **VJP:** equals `jacobian(...).T @ v` for random cotangents.
