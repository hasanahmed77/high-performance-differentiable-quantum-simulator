# Sprint 06: JAX backend II (batching, GPU) and v0.2.0

| | |
|---|---|
| **Status** | Planned: awaiting PO approval |
| **Started** | 2026-11-17 |
| **Milestone** | `Sprint 06 – JAX backend II (batching, GPU)` |
| **Sprint goal** | A user can batch circuits with `jax.vmap`, get adjoint gradients with depth-independent memory inside JAX programs, broadcast parameters through PennyLane, and run all of it on a GPU that CI tests. Release **v0.2.0**. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-907 | A hardened self-hosted GPU runner on the workstation, and `gpu.yml` | 3 | ADR-0007 | |
| QS-605 | `jax.vmap` batching over parameters (re-estimated from 5) | 3 | ADR-0007 | |
| QS-606 | Adjoint inside JAX as a `jax.custom_vjp`, opt-in | 5 | QS-605 | |
| QS-405 | One adjoint pass for many observables, proven on NumPy and JAX (re-estimated from 3) | 1 | QS-606 | |
| QS-506 | PennyLane parameter broadcasting, as one batched execution on the JAX backend | 3 | QS-605 | |
| QS-607 | The JAX backend on GPU: GPU-vs-CPU tolerance suite in `gpu.yml`, GPU docs | 3 | QS-907 | |
| QS-1101 | ADR: benchmark methodology (**stretch**, from Sprint 7) | 2 | n/a | |
| | **Total** | **20** (18 committed + 2 stretch) | | |

**Release:** v0.2.0 after the stories, through the existing pipeline (ADR-0007 D6).

**Capacity.** Mean throughput over six sprints is 21.5 points. The 18 committed points leave room for the release
work, which isn't pointed, and for QS-907's setup at the workstation. QS-1101 is the stretch: it's independent,
and Sprint 7's benchmarks need it first.

**Re-estimates (the spike, ADR-0007):**

- **QS-605, 5 → 3.** `jax.vmap` already works on Sprint 5's compiled circuits: identical results to a loop, one
  compilation. What's left is the error message, the tests, the docs and a timing, not new machinery.
- **QS-405, 3 → 1.** The NumPy adjoint has made one backward pass for many observables since Sprint 3 (#121); no
  test asserts it. On JAX it comes with QS-606's backward rule. The story becomes the test on both backends.

**Sequencing.** QS-907 goes first, so every later JAX PR gets a GPU run before merging (ADR-0007 D5). Then the
JAX stories in order: batching, adjoint (with QS-405), broadcasting. QS-607 is last, because it adds the GPU
suite to `gpu.yml`. QS-1101 can go in while a JAX PR waits on CI.

**Scope changes in this plan (for PO approval):**

- **GPU tests before merging are a manual `gpu.yml` dispatch on the PR's branch,** not a `pull_request`
  trigger. plan.md §4.5 forbids running PR code on the workstation, which the Sprint 5 retro's "Try" didn't take
  into account. A dispatch runs only branches of this repository, triggered by someone with write access.
- **The runner gets its own WSL distribution and rootless Docker,** stricter than plan.md §4.5 (ADR-0007 D5).
- **Adjoint in JAX is opt-in; backprop stays the default,** because `custom_vjp` rules out `jax.hessian`
  (ADR-0007 D2).
- **Sampling stays on the host** on every device (ADR-0007 D4). On-device sampling waits for Sprint 7's numbers.

## Test plan

| ID | Verification |
|---|---|
| QS-907 | `gpu.yml` triggers only on `workflow_dispatch`, `schedule` and `push` to `main` (a test reads the workflow); steps inside the job assert it isn't root, can't read `/mnt/c` and has no secrets; a green run on a branch and on `main` |
| QS-605 | **Differential:** `jax.vmap` values and gradients equal a loop on random circuits, both precisions; **trace count:** one compilation per structure, batched or not; a 2-D input raises the error that names `jax.vmap`; **timing:** batch speedup measured on the GPU and reported in the PR, including where it's 1× |
| QS-606 | `expval_fn(..., diff_method="adjoint")` equals backprop and `gradients.jacobian(..., method="adjoint")` on random circuits, both precisions, with shared parameters, under `jit` and `vmap`; `jax.hessian` raises the documented error; **memory:** XLA's temporary memory for the adjoint gradient is flat from 20 to 400 layers, while backprop's grows; `JaxBackend.adjoint_jacobian` passes the cross-backend suite on the new path |
| QS-405 | Counting gate applications: the Jacobian of 1 and of 8 observables both make one backward pass, on NumPy and on JAX |
| QS-506 | Broadcast QNodes (`qml.RX(np.array([...]))`) on `backend="jax"` equal `default.qubit`: values, `jax.grad`, `jax.jit`; one execution, not one per batch entry; adjoint and parameter-shift on broadcast tapes still expand; the NumPy backend still expands; `pl-device-test` unchanged |
| QS-607 | `gpu.yml` runs the JAX backend suite on the RTX 4060 and compares GPU against CPU at plan.md §6.3's tolerances; `check_gpu.py` passes; the GPU section of the docs is executed by a test on the CPU |
| QS-1101 | The ADR states warm-up, repeats, statistics and the hardware record, and Sprint 7's suite follows it |
| Sprint | **Integration:** a batch of H₂ VQE starting points, `jax.vmap`-ed with adjoint gradients in one jit-compiled step, on the GPU, all reaching the FCI energy. **Release:** v0.2.0 on PyPI, smoke-tested on three operating systems |

## Demo

A batch of training runs in one compiled step, with adjoint gradients, on the GPU:

```python
energy = expval_fn(ansatz, [H], diff_method="adjoint")          # adjoint memory inside JAX
step = jax.jit(jax.vmap(lambda x: x - 0.1 * jax.grad(lambda x: energy(x)[0])(x)))
xs = step(starting_points)                                       # 64 starting points at once
```

And PennyLane broadcasting:

```python
dev = qml.device("ketjet.qubit", wires=2, backend="jax")

@qml.qnode(dev, interface="jax")
def circuit(x):
    qml.RX(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

circuit(jnp.linspace(0, jnp.pi, 100))   # one batched execution, not 100
```

## Actions needed from the PO

1. **Approve ADR-0007**, the re-estimates and the scope changes above, by merging this PR.
2. **QS-907 needs you at the PC** for two steps I can't take alone:
   - registering the runner, which creates a credential for the repository, so I'll ask first;
   - turning on *Require approval for all outside collaborators* under Settings → Actions → General, a security
     setting that's yours to change.
   The dedicated WSL distribution needs the Ubuntu 24.04 image again (about 389 MB), since the installers were
   deleted after Sprint 5.
3. **Approve the v0.2.0 release** in the `pypi` environment at the end of the sprint, as for v0.1.0.
