# ADR-0007: Batching is `jax.vmap`, adjoint in JAX is opt-in, and GPU tests run on demand

| | |
|---|---|
| **Status** | Proposed (for PO approval in the Sprint 06 plan PR) |
| **Date** | 2026-11-17 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-605 (vmap), QS-606 (adjoint as `custom_vjp`), QS-405 (one adjoint pass for many observables), QS-607 (GPU), QS-506 (PennyLane broadcasting), QS-907 (GPU runner) |

## Context

ADR-0006 built the JAX backend so that a compiled circuit is a pure function of one parameter array, and left
three follow-ups for Sprint 6: batching (QS-605), adjoint memory inside JAX programs (QS-606), and GPU execution
with on-device sampling as an open question (QS-607). Sprint 6 also adds PennyLane's parameter broadcasting
(QS-506) and the GPU CI runner (QS-907). The Sprint 5 retrospective asked for GPU tests on pull requests, which
plan.md §4.5's runner security model forbids as written.

### What the Sprint 6 planning spike measured

On the RTX 4060 workstation, in the pinned `ketjet-gpu` image (JAX 0.12.0.1, `main` at `302ef00`):

| Question | Result |
|---|---|
| `jax.vmap` over a `compile_circuit` program (6 qubits, batch of 32) vs a Python loop | Max difference **0.0**, for values and for `jax.vmap(jax.grad(...))` |
| Compilations for that circuit across `vmap`, the loop, `vmap(grad)` and the gradient loop | **1** trace |
| `jax.custom_vjp` function under `jax.grad` | Works |
| The same function under `jax.jacfwd` or `jax.hessian` | **Fails:** `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` |
| Backprop's temporary memory from XLA's compile-time accounting, on the **CPU** (10 qubits, `RY` + `CNOT` layers) | **53, 204 and 807 states** for 5, 20 and 80 layers: it grows linearly with depth, as ADR-0006 D7 said |
| `vmap` vs a loop of single compiled calls, complex64, 4 layers | 12 qubits, batch 256: **7.1×** faster (18.9 vs 135.1 ms); 16 qubits, batch 64: **1.4×**; 20 qubits, batch 16: **1.0×** |

The timings are rough (5 repeats, one machine). They show the shape: batching pays off when one circuit is too
small to fill the GPU, and stops paying off when one circuit already does. Sprint 7's benchmarks measure it
properly.

What the code already does:

- The NumPy adjoint (`ketjet.adjoint`, Sprint 3) computes the Jacobian of `M` observables in **one** backward
  pass with `2 + M` states, so QS-405's acceptance criterion is met there. No test asserts it yet.
- `JaxBackend.adjoint_jacobian` uses `jax.jacrev`, so its memory grows with circuit depth (ADR-0006 D7).
- `ketjet.qubit` handles a broadcast tape with PennyLane's `broadcast_expand`, one execution per batch entry.
  Nothing tests broadcasting.

## Decision

### D1. Batching is `jax.vmap` over the compiled function, with no batched kernels

- `jax.vmap(run)` and `jax.vmap(energy)` are the batching API. A compiled circuit is already a pure function
  of one array (ADR-0006 D2), and the spike shows `vmap` gives identical results with one compilation.
- `CompiledCircuit.__call__` keeps rejecting a 2-D array, and its error message now names `jax.vmap`, so the
  common mistake leads to the fix.
- The documentation states the cost: a batch of `B` holds `B` states, so memory is `B × 2**n` amplitudes, and
  batching helps most when `n` is small.

| Option | Pros | Cons |
|---|---|---|
| **A. `jax.vmap` over the existing function (chosen)** | No new kernels; composes with `jit` and `grad`; already correct | Users learn one JAX idiom |
| B. A `batched=True` flag that adds a leading axis | One call for users | A second code path that does what `vmap` does |
| C. Hand-written batched kernels | Full control | Re-implements what XLA already generates |

### D2. Adjoint inside JAX is a `jax.custom_vjp`, and it's opt-in

- A new function builds a differentiable expectation-value function:
  `expval_fn(circuit, observables, *, dtype, diff_method="backprop" | "adjoint")`, returning `f(values) -> (M,)`.
  It's the JAX counterpart of `gradients.jacobian`'s `method` argument.
- With `"adjoint"`, `f` is a `jax.custom_vjp`. The forward pass keeps only the final state and the parameters.
  The backward pass runs the adjoint recursion of ADR-0004 in `jax.numpy`, undoing one gate at a time. A VJP with
  cotangent `v` uses the single observable `Σ v_m O_m` (ADR-0004 D6), so it is one pass with three states, for
  any `M`. Under `jax.jacrev`, JAX batches that pass over the `M` cotangents.
- **Backprop stays the default.** `custom_vjp` rules out forward mode, so `jax.jacfwd` and `jax.hessian` fail
  with `"adjoint"`. The docstring and docs say so and point to backprop for Hessians.
- `JaxBackend.adjoint_jacobian` switches from `jax.jacrev` to this path. That meets ADR-0004 D4's memory bound
  on JAX and closes ADR-0006 D7's caveat.
- **QS-405 lands here:** one test asserts one backward pass for many observables on NumPy (as built in Sprint 3),
  and the same on JAX through this rule.
- Memory is checked from XLA's own compile-time accounting (`lower(...).compile().memory_analysis()`), which works
  on the CPU in CI: the temporary memory of the adjoint gradient must not grow with depth, while backprop's does.

| Option | Pros | Cons |
|---|---|---|
| **A. `custom_vjp`, opt-in (chosen)** | Adjoint memory inside JAX programs; composes with `jit` and `vmap` | No forward mode, so no `hessian` |
| B. `custom_vjp` as the default | Low memory by default | Breaks `jax.hessian`, which works today (#139) |
| C. `custom_jvp` | Works with forward and reverse mode | Adjoint is a reverse-mode method; a JVP rule needs a forward pass per parameter |

### D3. PennyLane broadcasting runs as one batched execution on the JAX backend

- With `backend="jax"`, `ketjet.qubit` runs a broadcast tape as one `vmap`-ed execution: no
  `broadcast_expand`, for plain execution and for `diff_method="backprop"`.
- The NumPy backend keeps `broadcast_expand`. Batched NumPy kernels would be a second kernel family for a
  backend whose job is to be the plain oracle.
- Adjoint and parameter-shift on broadcast tapes keep `broadcast_expand` on both backends, as PennyLane's own
  devices do.
- Results must equal `default.qubit` on broadcast tapes: values, `jax.grad` and `jax.jit`.

### D4. GPU execution needs no new code; sampling stays on the host

- The JAX backend runs on whatever device JAX places it on. ketjet adds no `device` argument: JAX's own
  `jax.default_device` and `jax.device_put` apply, as for any JAX program.
- Contractions keep full float32 precision on GPUs (ADR-0006 D4, amended in #145).
- **Sampling stays on the host** (ADR-0006 D6). Only the measured wires' probabilities leave the device, and a
  seed gives the same bits on every backend and every device. On-device sampling with `jax.random` would give
  different bits per backend and break that property. It's revisited only if Sprint 7's benchmarks show host
  sampling is a bottleneck.
- GPU results must match the CPU within plan.md §6.3's tolerances.

| Option | Pros | Cons |
|---|---|---|
| **A. Host sampling (chosen)** | One seeded sampler for every backend; already built and tested | Copies the measured wires' probabilities to the host |
| B. `jax.random` on the device | No copy; samples stay on the GPU | A seed gives different bits per backend; a second sampler to test |

### D5. GPU tests: `gpu.yml` on a self-hosted runner, run on a PR's branch on demand

plan.md §4.5's security model stands: `gpu.yml` runs **only** on `workflow_dispatch`, `schedule` and `push` to
`main`, never on `pull_request`, so code from forks never reaches the workstation. This ADR adds:

- **GPU tests before merging, by dispatch.** `workflow_dispatch` can target any branch in this repository, and
  only users with write access can trigger it. A PR that changes the JAX backend or `environment/` gets a
  `gpu.yml` run on its branch before merging, linked in the PR. That answers the Sprint 5 retro without running
  fork code. It can't be a required check, since it doesn't run on PR events, so it's a rule in the PR template
  and the Definition of Done for GPU stories.
- **A dedicated WSL distribution for the runner,** separate from the PO's Ubuntu. Its `wsl.conf` turns off
  automounting of Windows drives and Windows interop, so a job can't read the PO's Windows files or start
  Windows programs.
- **Rootless Docker for the runner user**, so the runner never has root inside WSL. Each job runs in the pinned
  `ketjet-gpu` image with no secrets. The NVIDIA container toolkit supports rootless Docker.
- `schedule`: nightly. The workstation isn't always on, so the run waits until it is, and the sprint-level DoD
  for GPU sprints keeps requiring a green `gpu.yml` run before the sprint review.

| Option | Pros | Cons |
|---|---|---|
| **A. Dispatch on a branch, dedicated distro, rootless Docker (chosen)** | GPU tests before merge; fork code never runs; a job can't reach Windows or root | Manual trigger; one more WSL distribution to maintain |
| B. `gpu.yml` on `pull_request` from this repository only | Automatic | A workflow change in a PR from a fork can still target the runner; the security model forbids it |
| C. The runner in the PO's Ubuntu, Docker group | Least setup | The Docker group is root, and the PO's Windows files are mounted |

**Residual risk:** WSL2 distributions share one VM kernel, so a kernel exploit inside a job could cross distros.
The dedicated distro and rootless Docker remove every ordinary path. Only code from this repository's branches
runs, and only when someone with write access triggers it or it's on `main`.

### D6. v0.2.0 goes through the existing release pipeline

The QS-1205 pipeline (TestPyPI, smoke tests on three operating systems, PyPI after PO approval) releases v0.2.0
unchanged. The release includes the PennyLane floor fix from #141 (`pennylane>=0.44.1`).

## Consequences

- **Positive:**
  - Batching, adjoint memory and the GPU need little new machinery: `vmap` and JAX's device placement do most of
    the work, and the oracle stays plain NumPy.
  - Adjoint gradients inside JAX programs keep ADR-0004's memory bound, which matters at the 26–27 qubit range on
    the 8 GB GPU (plan.md §4.5).
  - GPU bugs like TF32 (#145) are caught before merging, not a day after.
- **Negative:**
  - `"adjoint"` in JAX has no forward mode: no `jax.hessian`.
  - Batching multiplies memory by the batch size and doesn't help large circuits.
  - GPU tests before merging rely on a manual dispatch and the workstation being on.
  - One more WSL distribution to set up and keep updated.
- **Follow-ups:**
  - On-device sampling, only if Sprint 7 shows host sampling is slow (D4).
  - Batched NumPy kernels aren't planned; the native backend (Sprints 8–9) decides its own batching.

## Validation

- **vmap:** batched values and gradients equal a loop on random circuits, both precisions; one trace per
  structure; a 2-D input to `CompiledCircuit` raises the error that names `jax.vmap`.
- **Adjoint in JAX:** `expval_fn(..., diff_method="adjoint")` equals backprop and `ketjet.gradients.jacobian`
  with `"adjoint"`, under `jit` and `vmap`; `jax.hessian` raises the documented error; XLA's temporary memory for
  the adjoint gradient is flat from depth 20 to depth 400, while backprop's grows.
- **One pass (QS-405):** counting gate applications, the Jacobian of 1 and of 8 observables both make one
  backward pass, on NumPy and JAX.
- **Broadcasting:** broadcast QNodes on `backend="jax"` equal `default.qubit` for values, `jax.grad` and
  `jax.jit`, and run as one execution; on the NumPy backend they still expand.
- **GPU:** `gpu.yml` runs the JAX test suite on the RTX 4060 and compares with the CPU at plan.md §6.3's
  tolerances; a green run on `main` before the sprint review.
- **Runner:** the job can't read `/mnt/c`, isn't root, and has no secrets. Each is checked by a step in
  `gpu.yml`.
