# Sprint 05: JAX backend I

| | |
|---|---|
| **Status** | Done: accepted by the PO by merging this review |
| **Review** | 2026-11-16 |
| **Started** | 2026-11-15 |
| **Milestone** | `Sprint 05 – JAX backend I` |
| **Sprint goal** | A user can run any ketjet circuit on a JAX backend that matches the NumPy oracle, compiles once per circuit structure, and is differentiable with `jax.grad`, natively and through PennyLane's JAX interface. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-601 | Spike and ADR-0006: the JAX backend design | 2 | n/a | #135 |
| QS-602 | `JaxBackend` passes the full cross-backend suite; `ketjet[jax]` extra; `pl-device-test` on the JAX backend | 5 | ADR-0006 | #136 |
| QS-603 | `compile_circuit`: one `jax.jit` compilation per circuit structure | 3 | QS-602 | #137 |
| QS-604 | `jax.grad` through the simulator (backprop) | 3 | QS-603 | #138 |
| QS-608 | PennyLane's JAX interface end to end: `backend="jax"`, `diff_method="backprop"`, `jax.jit` | 3 | QS-604 | #139 |
| QS-110 | Dependabot, CodeQL and pip-audit | 2 | n/a | #140 |
| QS-112 | Reproducible GPU workstation environment (WSL2 Ubuntu, CUDA 12, pinned image, setup guide) | 3 | n/a | #142, #143, #144, #146 |
| QS-507 | CI job for the minimum and latest supported PennyLane and JAX versions (**stretch**) | 2 | QS-608 | #141 |
| | **Total** | **23** (21 committed + 2 stretch) | | |

**Capacity.** Mean throughput over five sprints is 21.2 points, so the 21 points of plan.md §8 are committed.
QS-507 is a stretch goal. It moved up because the spike found a version boundary that CI should guard (below).

**Sequencing.** The JAX stories are serial: parity first, then compilation, then gradients, then PennyLane. QS-110
is independent and can go in while a JAX PR waits on CI. QS-112 comes last, as plan.md §8 says, so Sprint 6 starts
with a working GPU machine.

**Scope changes in this plan (approved by the PO in #135):**

- **The JAX range is `jax>=0.10`, not `≥0.12`** (ADR-0006 D1, revises plan.md D4). PennyLane 0.45.1, the latest
  release, fails under its JAX interface with JAX ≥0.11. With `≥0.12`, QS-608 couldn't be delivered until
  PennyLane ships a fix.
- **complex128 on JAX raises an error unless the user has enabled JAX's 64-bit mode** (ADR-0006 D5). ketjet doesn't
  switch a global JAX setting for the user.
- **`JaxBackend.adjoint_jacobian` uses reverse mode in this sprint** (ADR-0006 D7). Values are correct, but
  adjoint's depth-independent memory arrives on JAX with QS-606 in Sprint 6.
- **Backprop on the device arrives now**, with QS-604 and QS-608, not with QS-606 as ADR-0005 D4 said.
- **QS-507 widens** from PennyLane versions to PennyLane and JAX versions.

## Test plan

| ID | Verification |
|---|---|
| QS-602 | **Cross-backend:** the existing backend, measurement and sampling tests are parametrized over NumPy and JAX, in both precisions, with asymmetric states (Sprint 4 retro); **differential:** random circuits on JAX vs `default.qubit`; **registry parity:** every parametrized gate has a JAX matrix equal to the NumPy one; **precision:** complex128 without 64-bit mode raises the documented error; `import ketjet` loads no `jax` (subprocess); **conformance:** `pl-device-test --device-kwargs backend=jax`, exact and shot modes, in the required CI job |
| QS-603 | **Trace count:** many parameter sets on one structure trace once, a second structure traces once more, a bound circuit and its symbolic template share a compilation; the cache is bounded; compiled results equal the eager backend |
| QS-604 | `jax.grad` and `jax.jacrev` match the NumPy adjoint Jacobian and parameter-shift on random circuits, in both precisions; shared parameters; `jax.jit(jax.grad(...))` |
| QS-608 | QNodes on `backend="jax"` with `interface="jax"`: value, `jax.grad` and `jax.jit(jax.grad)` match `default.qubit` for `backprop`, `adjoint` and `parameter-shift`; decomposed gates included; `backprop` on the NumPy backend still raises PennyLane's error; runs on JAX 0.10.x |
| QS-110 | Dependabot opens grouped update PRs; CodeQL and `pip-audit` run in CI with no high-severity findings |
| QS-112 | `nvidia-smi` and `jax.devices()` show the RTX 4060 inside the pinned container; `docs/dev/gpu-workstation.md` reproduces the setup from a clean WSL2 install |
| QS-507 | A CI job on the minimum PennyLane and JAX versions and one on the latest of each that work together |
| Sprint | **Integration:** the H₂ VQE runs on the JAX backend with `jax.grad` inside a jit-compiled optimization step and reaches the FCI energy |

## Demo

A jit-compiled circuit against NumPy, and its gradient by backprop:

```python
import jax
import ketjet as kj
from ketjet.backends.jax_backend import compile_circuit

jax.config.update("jax_enable_x64", True)

run = compile_circuit(circuit)              # compiles once for this structure
state = run(params)                         # later calls reuse the compiled program
grad = jax.grad(lambda p: energy(run(p)))(params)
```

The spike's prototype ran 6.2× faster than NumPy at 10 qubits and 2.4× at 20 qubits after a 2–4 s compilation, on
the Mac's CPU.

## Actions needed from the PO

1. **Approve ADR-0006**, including the scope changes above, by merging this PR. Backend code starts after that.
2. **QS-112 needs you at the Windows PC.** I'll write the setup guide and the container definition. Running the
   WSL2 and NVIDIA driver steps, and confirming the two checks, has to happen on that machine.

## Sprint review

**Goal met? Yes.** Any ketjet circuit runs on `JaxBackend` and matches the NumPy oracle in both precisions. It
compiles once per circuit structure, and `jax.grad` differentiates it, natively and through PennyLane's JAX
interface with `diff_method="backprop"`. The sprint's integration test, the H₂ VQE with `jax.grad` inside a
jit-compiled step, reaches the FCI energy to 1e-12. The GPU workstation is set up, so Sprint 6 starts unblocked.

| ID | Outcome |
|---|---|
| QS-601 | ✅ Done (#135): ADR-0006 approved with the sprint's scope changes; D4 amended once, with PO approval (full float32 precision, #145) |
| QS-602 | ✅ Done (#136): `JaxBackend` passes the cross-backend suite in both precisions; `ketjet[jax]` extra; `pl-device-test` on the JAX backend from the first PR |
| QS-603 | ✅ Done (#137): `compile_circuit` traces once per structure; a bounded 128-entry cache |
| QS-604 | ✅ Done (#138): backprop with `jax.grad`, equal to the adjoint and parameter-shift Jacobians; `parameter_map` for shared symbols |
| QS-608 | ✅ Done (#139): `backend="jax"` with `diff_method="backprop"` under PennyLane's JAX interface, including `jax.jit` |
| QS-110 | ✅ Done (#140): pip-audit and CodeQL gate every PR; Dependabot configured; Dependabot alerts and security updates enabled by the PO |
| QS-112 | ✅ Done (#142, #143, #144, #146): pinned GPU image and setup guide; `check_gpu.py` passes on the RTX 4060; setup recorded |
| QS-507 | ✅ Done, stretch (#141): a required minimum-versions job and a daily latest-versions workflow. The minimum job found a bug in v0.1.0's PennyLane floor |

### Test results (sprint-level DoD, plan.md §5.3)

| Suite | Result |
|---|---|
| Unit | ✅ the protocol tests on every backend; registry parity; precision rules; traced parameters; trace counts and the bounded cache; `parameter_map`; import hygiene (`import ketjet` loads no JAX) |
| Differential | ✅ random circuits over every gate, JAX vs the NumPy oracle and `default.qubit`, both precisions (worst complex128 difference 7.1e-15); `jax.jacrev` vs adjoint and parameter-shift on 6 seeds × 2 precisions with shared symbols |
| Integration | ✅ the H₂ VQE with `jax.grad` in a jit-compiled step reaches the FCI energy to 1e-12 (#138); QNodes under `interface="jax"` equal `default.qubit` for backprop, adjoint and parameter-shift, under `jax.jit` too (#139) |
| **Conformance** | ✅ `pl-device-test` on the JAX backend: exact 394 passed / 0 failed (8 autograd-backprop tests deselected, with the reason documented); 10,000 shots 374 / 60 / 0. NumPy backend unchanged |
| Compatibility | ✅ minimum supported versions (Python 3.12, numpy 2.1.0, PennyLane 0.44.1, JAX 0.10.0): 1,785 passed, a required job. Latest versions: 1,776 passed, daily and on PRs |
| **GPU** | ✅ on the RTX 4060: `check_gpu.py` passes (complex128 1.9e-17, complex64 2.6e-8 vs NumPy); the JAX backend tests pass 153 of 153 with #145 (135 of 153 before) |
| Regression | ✅ every contraction of a compiled circuit and its gradient asks for full precision (a jaxpr test that runs on the CPU, #145); both `nvidia-smi` header formats (#144); the PennyLane 0.44.1 floor (#141) |
| Security | ✅ pip-audit: no known vulnerabilities in either resolution of `uv.lock`; CodeQL fails on any finding of severity ≥7.0 |
| Mutation checks | ✅ 47 injected bugs across 4 PRs, all caught. Two survived a first draft (#136), and each led to a new test |
| Benchmarks | n/a: Sprint 7 |
| Coverage | ✅ 100% (1,786 passed, 2 skipped: the JAX 0.10-only cases, which run in the conformance job) |
| Docs | ✅ ADR-0006; Backends sections in Getting started and on the PennyLane page; API reference; the GPU workstation guide with its setup record; `mkdocs build --strict` |

**Bugs found:**

- **v0.1.0 allows a PennyLane that can't be imported** (found by the minimum-versions job, #141). Its extra says
  `pennylane>=0.44`, and PennyLane 0.44.0 now fails on import with `ModuleNotFoundError: gast`, because a
  dependency dropped it. A default install picks 0.45.1 and works. The floor is `>=0.44.1` on `main`, for the next
  release.
- **complex64 on the GPU drifted from the oracle by up to 8e-3** (found on the workstation, #146; fixed in #145).
  XLA computes float32 products in TF32 by default on newer NVIDIA GPUs, and 18 of 153 JAX tests failed there. The
  code had merged in #136. No CPU test could see it, and the new jaxpr test now catches it on the CPU.
- **`check_gpu.py` skipped the driver check** on driver 610.88, whose header says `CUDA UMD Version` (#144).
- **PennyLane 0.44's 3-qubit `QubitUnitary` decomposition is accurate to about 4.5e-8**, against 4e-14 on 0.45.1
  (#141). It's PennyLane's limit, not ketjet's: the test's tolerance is scoped to 0.44, and `docs/pennylane.md`
  states it.

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 23 of 23 points (21 committed and the 2-point stretch), 8 of 8 stories (Sprint 0: 16, Sprint 1: 18, Sprint 2: 21, Sprint 3: 23, Sprint 4: 28) |
| Cycle time | Median 44 minutes from opening a story's PR to merging it (6 minutes to 2 h 52 min). QS-112 needed three follow-up PRs after the workstation run |
| Sprint duration | 2026-11-15 to 2026-11-16 |
| WIP limits | Held for stories: one story PR in review at a time. QS-112's three follow-ups were open together |
| Escaped defects | 1 released: v0.1.0's PennyLane floor (above). Two more reached `main` and were fixed before any release (TF32, `check_gpu.py`) |
| CI failures before merge | 0 caused by code. #144–#146 showed a false red **CI success** from runs cancelled when labels were added at PR creation, and one macOS job on #146 never got a GitHub runner |
| ADR amendments | 1 (ADR-0006 D4), approved by the PO before merging |

### Demo

A jit-compiled gradient step, from #138:

```python
run, to_slots = compile_circuit(ansatz), parameter_map(ansatz)
energy = lambda x: expectation(run(to_slots(x)), H)
step = jax.jit(lambda x: x - 0.5 * jax.grad(energy)(x))   # compiles the circuit once
```

And ketjet on the GPU, from `docker run --rm --gpus all ketjet-gpu` on the workstation (16 qubits, 8 layers,
136 parameters):

```text
== jax.devices()
   gpu:0 NVIDIA GeForce RTX 4060
== ketjet on gpu: 16 qubits, 8 layers
   complex128: first call 5.43 s, then 64.15 ms; max |state - NumPy| = 1.9e-17 ok
   complex64: first call 4.51 s, then 10.79 ms; max |state - NumPy| = 2.6e-08 ok
   jax.jit(jax.grad): 136 parameters, |grad| = 0.0404
```

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Test on the real hardware.** The first GPU run found two bugs that CPU-only CI couldn't, one of them a silently wrong complex64 result | **GPU bugs surfaced a day after they merged.** The TF32 bug sat on `main` from #136 until the workstation run. GPU tests should run on pull requests | Run the JAX tests on a GPU runner from Sprint 6's first PR (QS-907) |
| **Test both ends of every dependency range.** The minimum-versions job found a broken floor in the released v0.1.0 on its first run | **Labels at PR creation gave false red checks.** Each label starts a CI run that cancels the previous one, and a cancelled run reports a failed **CI success**. PRs now open without labels, but adding `skip-changelog` later still does it | Make a cancelled run's **CI success** not report failure in `ci.yml` (a PO decision) |
| **Run conformance from the first PR.** `pl-device-test` ran on the JAX backend from #136, as the Sprint 4 retro asked, and nothing turned up late | **Flag ADR deviations before merging, every time.** The D4 amendment was flagged in #145 and approved before it merged; keep it that way | Run `mutmut` nightly, and the doctests in the Linux image, which this PC can now build (both carried over) |

## Forecast update

Throughput over six sprints: 16, 18, 21, 23, 28 and 23 points (mean 21.5). 192 points remain across Sprints 6–16,
which is about 8.9 sprints at the current pace, against the 11 planned. Sprint 6 finishes the JAX backend (batching,
adjoint inside JAX, GPU) and releases **v0.2.0**, on the GPU workstation.
