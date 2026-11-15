# Sprint 05: JAX backend I

| | |
|---|---|
| **Status** | In progress |
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
| QS-112 | Reproducible GPU workstation environment (WSL2 Ubuntu, CUDA 12, pinned image, setup guide) | 3 | n/a | |
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
