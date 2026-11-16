# ADR-0006: The JAX backend compiles a circuit's structure once and takes its parameters as an array

| | |
|---|---|
| **Status** | Accepted (approved by the PO on 2026-11-15 in #135; D4 amended on 2026-11-16 in #145) |
| **Date** | 2026-11-15 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-601 (this spike), QS-602, QS-603, QS-604, QS-608; later QS-605 (vmap), QS-606 (adjoint as `custom_vjp`), QS-607 (GPU), QS-507 (version range) |

## Context

The NumPy backend is the correctness oracle. It's deliberately plain, so it can't be compiled, traced by an autodiff
framework or run on a GPU. The JAX backend adds those three things behind the same `Backend` protocol (ADR-0001 D4).
It has to:

- return the same results as the NumPy oracle, in both precisions (QS-602);
- compile a circuit once and rerun it with new parameter values without recompiling (QS-603);
- be differentiable by `jax.grad`, which gives ketjet backprop for the first time (QS-604, deferred from
  ADR-0005 D4);
- work under PennyLane's JAX interface (QS-608);
- keep `import ketjet` free of JAX, as it's free of PennyLane.

Three properties of the current code shape the design:

- `Operation` stores each parameter as a validated Python `float` (ADR-0002 D7). A JAX tracer isn't a float, so
  operations can't carry traced values, and a value stored in an operation would be compiled in as a constant.
- Gate matrices are NumPy functions (`GateSpec.matrix_fn`). `jax.grad` can't see through them.
- `jax.jit` compiles one program per distinct input *structure*. Anything that isn't an array argument is part of
  the structure.

### What the QS-601 spike measured

On the Mac M2 (CPU), with a prototype of the design below:

| Question | Result |
|---|---|
| Parity with the NumPy oracle (5 qubits, 63 gates incl. `CRZ`, `Toffoli`) | max abs difference 2.4e-16 (complex128), 1.4e-7 (complex64) |
| Recompiles for 20 new parameter sets on the same structure | **0** |
| `jax.grad` vs the NumPy adjoint Jacobian (33 parameters) | max abs difference 2.0e-16 |
| Speed after compiling, vs NumPy | 6.2× (10 qubits, 620 gates), 1.8× (16 qubits, 980 gates), 2.4× (20 qubits, 610 gates) |
| First-call compile time | 2.4 s, 4.4 s and 3.9 s for those three circuits |
| complex128 requested while JAX's 64-bit mode is off | JAX returns **complex64** with a warning |

Version facts, checked against PyPI and by running both libraries:

| Fact | Evidence |
|---|---|
| The latest JAX is 0.12.0.1. It needs Python ≥3.12 and ships wheels for Python 3.12–3.15 on Linux, macOS (arm64) and Windows | PyPI file list |
| JAX 0.10.2 ships wheels for Python 3.11–3.14, not 3.15 | PyPI file list |
| **PennyLane 0.45.1 (the latest release) fails under its JAX interface with JAX ≥0.11.** It calls `jax.core.is_concrete`, which JAX deprecated in 0.10 and removed in 0.11 | `AttributeError` reproduced on 0.11.2 and 0.12.0.1 |
| PennyLane 0.45.1 works with JAX 0.9.2 and 0.10.2 (with a deprecation warning on 0.10) | QNode with `jax.grad` and `jax.jit` on `default.qubit` |
| `ketjet.qubit` v0.1.0 already runs under `interface="jax"` with adjoint and parameter-shift, including inside `jax.jit` | PennyLane wraps the device in a host callback |
| `pl-device-test` with JAX 0.10.2 installed: 389 passed, 0 failed | 9 more tests than without JAX (380) |

## Decision

### D1. JAX is an optional extra, and the supported range is `jax>=0.10`

- `pip install "ketjet[jax]"`. `import ketjet` and `import ketjet.backends` never import JAX; a subprocess test
  checks this, as one does for PennyLane. The backend lives in `ketjet.backends.jax_backend`.
- The range is `jax>=0.10` with no upper bound. ketjet's own JAX code uses only stable public API (`jax.numpy`,
  `jit`, `grad`), and the prototype ran unchanged on 0.10.2 and 0.12.0.1.
- **PennyLane with JAX needs `jax<0.11`** until PennyLane releases a fix. That is PennyLane's limit, not ketjet's,
  so it isn't written into ketjet's metadata. It's documented on the PennyLane page, and CI pins it where
  PennyLane and JAX are tested together.
- CI tests both ends: the main matrix uses the latest JAX, and the PennyLane-JAX tests run on JAX 0.10.x.
- **This revises plan.md D4**, which assumed JAX ≥0.12. On Python 3.15 the JAX backend is tested (JAX 0.12), and
  the PennyLane-JAX tests are skipped, as every PennyLane test already is there.

| Option | Pros | Cons |
|---|---|---|
| **A. `jax>=0.10`, test both ends (chosen)** | PennyLane's JAX interface works today; users on the latest JAX aren't blocked | Two JAX versions in CI |
| B. `jax>=0.12` as plan.md assumed | One version to test | QS-608 can't be delivered until PennyLane ships a fix, on a date we don't control |
| C. `jax>=0.10,<0.11` | One version to test | Blocks Python 3.15 and every user on a current JAX |

### D2. A circuit is split into static structure and a parameter array

`ketjet.backends.jax_backend.compile_circuit(circuit, dtype)` returns a pure function `run(params) -> state`:

- **Structure** is everything that isn't a parameter value: the number of wires, and for each operation its gate
  name, target wires, control wires and control values. It's plain Python data, fixed when the function is built.
- **Parameters** are one flat real JAX array, in operation order: the `k`-th entry is the `k`-th parameter slot of
  the circuit. Numeric and symbolic parameters are both slots, so a bound circuit and its symbolic template share
  one structure.
- `run` is built by unrolling the gate list in Python. JAX records one array program with no Python control flow
  left in it.

This is the "pure functions, static wires" rule in the story's title. Wires can't be traced values in any case: an
array's axis has to be known when the program is compiled.

| Option | Pros | Cons |
|---|---|---|
| **A. Structure static, parameters one array (chosen)** | One compilation per structure; `grad` and (later) `vmap` apply to a single argument | Compile time grows with the gate count, since the loop is unrolled |
| B. Let `Operation` hold traced values | No second code path | Breaks ADR-0002 D7's validation; values held in operations compile in as constants, so every new value recompiles |
| C. `lax.scan` over a uniform gate table | Compile time independent of depth | Every gate must be padded to one shape and applied by dynamic indexing, which gives up the kernel structure and is slower per gate |

### D3. Gate matrices get a JAX twin for parametrized gates only

- A registry in the JAX backend maps each **parametrized** gate name (`RX`, `RY`, `RZ`, `PhaseShift`, `Rot`, `U3`)
  to a `jax.numpy` matrix builder.
- Fixed gates reuse the NumPy registry's constant matrices.
- A test iterates over `ketjet.gates.GATES` and fails if a parametrized gate has no JAX builder, or if a builder
  differs from the NumPy matrix on random parameters. A new gate can't be added to one registry and forgotten in
  the other.

| Option | Pros | Cons |
|---|---|---|
| **A. JAX builders for parametrized gates, checked against the oracle (chosen)** | The oracle isn't touched; six small functions | Two definitions of six matrices, held together by a test |
| B. Make `gates.py` generic over the array library | One definition | Puts an abstraction into the oracle, whose value is that it's plain NumPy |
| C. Build each matrix from its generator with `expm` | No per-gate code | Slower, and less accurate than closed forms |

### D4. Kernels: dense contraction, with controls applied on a slice

- The state is the flat `2**n` array of ADR-0001, viewed as a rank-`n` tensor inside a kernel.
- A gate's base matrix is contracted with its target axes (`jnp.tensordot`, then `jnp.moveaxis`): the same
  formulation as the oracle's `apply_matrix`.
- **Controls** select the control-satisfied slice with a static index, apply the base gate there and write it
  back. The `2**(k+c)` controlled matrix is never built, which is the reason ADR-0002 D2 keeps controls as
  structure.
- Diagonal gates multiply by a broadcast diagonal instead of contracting.
- No permutation kernel yet. XLA already compiles a contraction with a constant permutation matrix well, and
  Sprint 7's benchmarks decide whether a dedicated kernel is worth having.
- **Every contraction runs at full float32 precision** (`jax.lax.Precision.HIGHEST`): gate contractions, the
  matrix products that build gates, and the inner products in `expval` and `var`. XLA's default precision computes
  float32 matrix products in TF32 on NVIDIA GPUs from Ampere on. That keeps a 10-bit mantissa, and on the RTX 4060
  complex64 states and gradients drifted from the oracle by up to 8e-3. The precision is set on each operation,
  not with the global `jax_default_matmul_precision`, which would change the user's JAX program (as D5 rules
  out). Nothing changes on the CPU or in complex128, and at 16 qubits on the RTX 4060 the run time didn't change.
  (Amended in QS-607 by #145, approved by the PO on 2026-11-16; the first draft didn't set a precision.)

### D5. Precision is explicit: complex128 needs JAX's 64-bit mode, and ketjet never switches it on

JAX computes in 32 bits unless `jax_enable_x64` is set. With it off, asking for complex128 returns complex64 and a
warning. ADR-0001 says a state's precision is fixed at allocation and never changes silently.

- The backend's default precision stays complex128, as the protocol says.
- If complex128 is requested while 64-bit mode is off, the backend raises an error that names the fix:
  `jax.config.update("jax_enable_x64", True)`.
- complex64 works in either mode.
- The test suite turns 64-bit mode on in `conftest.py`.

| Option | Pros | Cons |
|---|---|---|
| **A. Raise a clear error (chosen)** | No silent precision loss; the user's JAX configuration stays theirs | One extra line for the user |
| B. Turn 64-bit mode on when the backend is imported | Works out of the box | A global side effect on the user's whole JAX program, from an import |
| C. Follow JAX's mode (complex64 when it's off) | Works out of the box | The same call returns different precision depending on global state |

### D6. Two layers: the `Backend` protocol for parity, the compiled function for speed and gradients

- **`JaxBackend`** implements the full `Backend` protocol with JAX arrays as states. Each method is a pure
  function of JAX arrays, so it runs eagerly and passes the cross-backend suite (QS-602). `Circuit.state(...,
  backend=JaxBackend())` works unchanged.
- **`compile_circuit`** (D2) is the fast path. It's cached on `(structure, dtype)` in a bounded least-recently-used
  cache, so a second circuit with the same structure reuses the compiled program (QS-603). Both layers share the
  same kernels.
- `probs`, `expval` and `var` are implemented in `jax.numpy`, so they can be differentiated. Results that leave
  the protocol are converted to NumPy types, which keeps the protocol's return types the same on every backend.
- **Sampling stays on the host.** `sample` computes probabilities in JAX, copies them to NumPy and uses ketjet's
  seeded sampler. A seed therefore gives the same bits on every backend. `jax.random` on the device is a
  Sprint 6 question (QS-607).

### D7. Differentiation: backprop now, adjoint inside JAX in Sprint 6

- **Backprop (QS-604):** `jax.grad` of a function built on `compile_circuit` is the gradient. No ketjet code is
  involved beyond being traceable. Its memory grows with circuit depth, since reverse mode stores the forward
  pass. That's the trade-off adjoint was built to avoid (ADR-0004), and the documentation says so.
- **`JaxBackend.adjoint_jacobian`** returns the Jacobian by reverse mode (`jax.jacrev`) in Sprint 5. The values and
  shape are those the protocol specifies, but **the memory bound of ADR-0004 D4 isn't met on this backend yet**.
  QS-606 (Sprint 6) replaces it with the adjoint recursion as a `jax.custom_vjp`. Until then the adjoint memory
  test runs on the NumPy backend only, and the JAX method's docstring states the limit.
- Parameter-shift and finite differences call the backend like any other and need nothing new.

### D8. PennyLane: one device, a `backend` argument, backprop when the backend is JAX

- `qml.device("ketjet.qubit", wires=n, backend="jax")`. The argument accepts `"numpy"` (default), `"jax"` or a
  `Backend` instance, which it already takes.
- With the JAX backend and a JAX interface, `supports_derivatives` reports `backprop` as supported. `execute`
  translates the tape 1:1 as before (ADR-0005 D2), except that each parameter becomes a slot of the parameter
  array instead of a float inside an `Operation`. The 1:1 rule is what makes this a small change: tape parameter
  `k` is still exactly one slot.
- `diff_method="adjoint"` and `"parameter-shift"` behave as on the NumPy backend, including under `jax.jit`.
- **This updates ADR-0005 D4**, which tied backprop to QS-606. Backprop arrives in Sprint 5 with QS-604 and
  QS-608; QS-606 adds adjoint memory inside JAX programs.
- **Conformance from the first PR** (Sprint 4 retro): the conformance job runs
  `pl-device-test --device=ketjet.qubit --device-kwargs backend=jax` from QS-602 on, next to the NumPy run.

## Consequences

- **Positive:**
  - Repeated runs of one circuit shape, which is what every optimization loop does, cost one compilation.
  - Backprop works, natively and through PennyLane, without changing the IR or the oracle.
  - `vmap` (QS-605) and GPU execution (QS-607) need no redesign: the compiled function is already a pure function
    of one array.
- **Negative:**
  - The first call pays 2–4 s of compilation for circuits of 600–1,000 gates, and compile time grows with the
    gate count. A circuit that runs once is faster on NumPy. The documentation says when to use which.
  - Six gate matrices are defined twice (D3).
  - The JAX backend's `adjoint_jacobian` doesn't have adjoint's memory bound until QS-606 (D7).
  - PennyLane users are limited to JAX 0.10.x until PennyLane fixes its JAX interface (D1).
- **Follow-ups:**
  - `vmap` batching (QS-605) and the adjoint `custom_vjp` (QS-606), Sprint 6.
  - GPU execution and on-device sampling (QS-607), Sprint 6.
  - Lift the JAX pin in the PennyLane CI jobs when PennyLane supports JAX ≥0.11 (QS-507).
  - Revisit a permutation kernel and compile time with Sprint 7's benchmarks.

## Validation

- **Parity:** the cross-backend suite runs every backend test against NumPy and JAX at the standard tolerances
  (plan.md §6.3), in both precisions.
- **No recompiles:** a trace-count test runs one structure with many parameter sets and asserts exactly one trace,
  and a second structure asserts exactly one more.
- **Gradients:** `jax.grad` equals the NumPy adjoint Jacobian and parameter-shift on random circuits.
- **Registry parity:** every parametrized gate has a JAX builder equal to its NumPy matrix.
- **Precision:** complex128 with 64-bit mode off raises the documented error; a complex64 state never becomes
  complex128.
- **Contraction precision:** a test walks the jaxpr of a compiled circuit and of its gradient and checks that
  every contraction asks for `HIGHEST`. It runs on the CPU, so CI catches a regression without a GPU (D4).
- **PennyLane:** a QNode on `backend="jax"` with `interface="jax"` gives the same value, `jax.grad` and
  `jax.jit(jax.grad)` as `default.qubit`; `pl-device-test` passes on the JAX backend.
- **Import hygiene:** `import ketjet` in a fresh interpreter loads neither `jax` nor `pennylane`.
