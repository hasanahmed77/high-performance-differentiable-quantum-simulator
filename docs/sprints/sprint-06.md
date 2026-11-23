# Sprint 06: JAX backend II (batching, GPU) and v0.2.0

| | |
|---|---|
| **Status** | Stories done, reviewed in this report; v0.2.0 not released yet. Plan approved by the PO in #149 |
| **Finished** | 2026-11-24 (stories) |
| **Started** | 2026-11-17 |
| **Milestone** | `Sprint 06 – JAX backend II (batching, GPU)` |
| **Sprint goal** | A user can batch circuits with `jax.vmap`, get adjoint gradients with depth-independent memory inside JAX programs, broadcast parameters through PennyLane, and run all of it on a GPU, tested by hand before every GPU PR (ADR-0007 D5, amended). Release **v0.2.0**. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| ~~QS-907~~ | ~~A hardened self-hosted GPU runner on the workstation, and `gpu.yml`~~ **Dropped by the PO on 2026-11-18** (ADR-0007 D5, amended) | ~~3~~ 0 | ADR-0007 | #154 |
| QS-605 | `jax.vmap` batching over parameters (re-estimated from 5) | 3 | ADR-0007 | #150 |
| QS-606 | Adjoint inside JAX as a `jax.custom_vjp`, opt-in | 5 | QS-605 | #151 |
| QS-405 | One adjoint pass for many observables, proven on NumPy and JAX (re-estimated from 3) | 1 | QS-606 | #151 |
| QS-506 | PennyLane parameter broadcasting, as one batched execution on the JAX backend | 3 | QS-605 | #152 |
| QS-607 | The JAX backend on GPU: GPU-vs-CPU tolerance suite, run by hand, GPU docs | 3 | n/a | #153 |
| QS-1101 | ADR: benchmark methodology (**stretch**, from Sprint 7) | 2 | n/a | #155 |
| | **Total** | **17** (15 committed + 2 stretch; was 18 + 2 before QS-907 was dropped) | | |

**Release:** v0.2.0 after the stories, through the existing pipeline (ADR-0007 D6).

**Capacity.** Mean throughput over six sprints is 21.5 points. The 18 committed points leave room for the release
work, which isn't pointed, and for QS-907's setup at the workstation. QS-1101 is the stretch: it's independent,
and Sprint 7's benchmarks need it first.

**Re-estimates (the spike, ADR-0007):**

- **QS-605, 5 → 3.** `jax.vmap` already works on Sprint 5's compiled circuits: identical results to a loop, one
  compilation. What's left is the error message, the tests, the docs and a timing, not new machinery.
- **QS-405, 3 → 1.** The NumPy adjoint has made one backward pass for many observables since Sprint 3 (#121); no
  test asserts it. On JAX it comes with QS-606's backward rule. The story becomes the test on both backends.

**GPU tests stay manual (PO decision, 2026-11-18).** The PO doesn't want the workstation reachable from GitHub
Actions, so QS-907 is dropped and there is no `gpu.yml` (ADR-0007 D5, amended). Every PR that changes the JAX
backend or `environment/` gets the GPU suite and `check_gpu.py` run by hand in the image, with the output pasted
into the PR, and the same run on `main` goes into this sprint's review.

**Sequencing.** *Changed with PO approval on 2026-11-17: QS-605 goes first, and QS-907 waits until the PO is at the
workstation; until then GPU checks run by hand in the image and are pasted into each PR.* QS-907 was planned first, so every later JAX PR gets a GPU run before merging (ADR-0007 D5). Then the
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
| ~~QS-907~~ | Dropped. ~~`gpu.yml` triggers only on `workflow_dispatch`, `schedule` and `push` to `main` (a test reads the workflow); steps inside the job assert it isn't root, can't read `/mnt/c` and has no secrets; a green run on a branch and on `main`~~ |
| QS-605 | **Differential:** `jax.vmap` values and gradients equal a loop on random circuits, both precisions; **trace count:** one compilation per structure, batched or not; a 2-D input raises the error that names `jax.vmap`; **timing:** batch speedup measured on the GPU and reported in the PR, including where it's 1× |
| QS-606 | `expval_fn(..., diff_method="adjoint")` equals backprop and `gradients.jacobian(..., method="adjoint")` on random circuits, both precisions, with shared parameters, under `jit` and `vmap`; `jax.hessian` raises the documented error; **memory:** XLA's temporary memory for the adjoint gradient is flat from 20 to 400 layers, while backprop's grows; `JaxBackend.adjoint_jacobian` passes the cross-backend suite on the new path |
| QS-405 | Counting gate applications: the Jacobian of 1 and of 8 observables both make one backward pass, on NumPy and on JAX |
| QS-506 | Broadcast QNodes (`qml.RX(np.array([...]))`) on `backend="jax"` equal `default.qubit`: values, `jax.grad`, `jax.jit`; one execution, not one per batch entry; adjoint and parameter-shift on broadcast tapes still expand; the NumPy backend still expands; `pl-device-test` unchanged |
| QS-607 | The GPU suite, run by hand, runs the JAX backend on the RTX 4060 and compares GPU against CPU at plan.md §6.3's tolerances; `check_gpu.py` passes; the GPU section of the docs is executed by a test on the CPU |
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
2. ~~**QS-907 needs you at the PC** for two steps I can't take alone:~~ Not needed: QS-907 was dropped on 2026-11-18.
   - registering the runner, which creates a credential for the repository, so I'll ask first;
   - turning on *Require approval for all outside collaborators* under Settings → Actions → General, a security
     setting that's yours to change.
   The dedicated WSL distribution needs the Ubuntu 24.04 image again (about 389 MB), since the installers were
   deleted after Sprint 5.
3. **Approve the v0.2.0 release** in the `pypi` environment at the end of the sprint, as for v0.1.0.

## Sprint review

**Goal met? Yes for the stories; the release is still to come.** A user can batch circuits with `jax.vmap`, get
adjoint gradients inside JAX whose memory doesn't depend on circuit depth, broadcast parameters through PennyLane,
and run all of it on a GPU. The sprint's integration run, 64 H₂ VQE starting points in one jit-compiled `vmap` step
with adjoint gradients on the RTX 4060, reached the FCI energy on all 64. **v0.2.0 isn't released yet:** it follows
this review and needs the PO's approval in the `pypi` environment.

| ID | Outcome |
|---|---|
| QS-605 | ✅ Done (#150): `jax.vmap` over compiled circuits, one compilation, and an error that names `jax.vmap` for a batch passed directly |
| QS-606 | ✅ Done (#151): `expval_fn` with `diff_method="adjoint"`; `JaxBackend.adjoint_jacobian` on the new path. Three deviations, approved by the PO: `jax.hessian` works, an `optimization_barrier` per gate is needed, and JAX 0.10 isn't fully flat. **Memory was measured over fewer layers than planned** (see below) |
| QS-405 | ✅ Done (#151): one backward pass for 1 and for 8 observables, on NumPy and JAX |
| QS-506 | ✅ Done (#152): broadcast tapes run as one batched execution on JAX, executed or under backprop; everything else still splits |
| QS-607 | ✅ Done (#153), **with changed scope:** a GPU-vs-CPU suite run by hand, not in `gpu.yml`; a GPU section in Getting started, executed by a test on the CPU |
| QS-1101 | ✅ Done, stretch (#155): ADR-0008, benchmark methodology, based on a timing-noise spike. The regression threshold starts at 10%, not 5%, with PO approval |
| QS-907 | ❌ **Dropped by the PO** (#154): the workstation is never reachable from GitHub Actions. GPU tests run by hand (ADR-0007 D5, amended) |

### Test results (sprint-level DoD, plan.md §5.3)

Run on `main` at `9ab47ec`, in the pinned `ketjet-gpu` image on the workstation, except where CI is named.

| Suite | Result |
|---|---|
| Full CPU suite | ✅ 1,866 passed, 3 skipped (the GPU module, and the two JAX 0.10-only modules, which run in the conformance job) |
| **GPU suite** | ✅ 44 of 44 on the RTX 4060 (JAX 0.12.0.1, driver 610.88): states, probabilities, expectation values, backprop and adjoint Jacobians, `vmap`, a 20-qubit circuit, `ketjet.qubit` and its seeded samples, adjoint memory at 18 qubits |
| JAX tests on the GPU | ✅ 209 passed, 1 skipped (the CPU's 12-qubit memory test; the GPU suite measures memory at 18 qubits) |
| `check_gpu.py` | ✅ All checks passed; complex128 1.9e-17 and complex64 2.4e-8 from the NumPy oracle at 16 qubits |
| **Integration** | ✅ 64 H₂ VQE starts, `jax.jit(jax.vmap(...))` with adjoint gradients on the GPU: 64 of 64 reach the FCI energy in 41 steps, worst error 2.7e-12 |
| **Conformance** | ✅ `pl-device-test` in CI. Counts from #152, on JAX 0.10 and PennyLane 0.45: JAX exact 394 passed, 32 skipped, 8 deselected, 9 xfailed; NumPy exact 389 passed, 45 skipped, 9 xfailed |
| Compatibility | ✅ the minimum-versions job (NumPy 2.1, PennyLane 0.44.1, JAX 0.10.0) and the daily latest-versions workflow |
| Security | ✅ pip-audit and CodeQL on `main` |
| Mutation checks | Not run this sprint |
| Benchmarks | n/a: the suite is Sprint 7. ADR-0008 sets its method |
| Coverage | ✅ 100% |
| Docs | ✅ ADR-0007 (amended), ADR-0008, Getting started (adjoint, batching, GPU), the PennyLane page's Broadcasting section; `mkdocs build --strict` |

**Where the sprint fell short of its plan:**

- **Adjoint memory was measured over fewer layers than planned.** The test plan said 20 to 400 layers. The CPU test
  covers 4 to 12 layers at 12 qubits, and spikes reached 16 on the CPU and 32 on the GPU, because deep circuits take
  minutes to compile and one uncapped 80-layer compile froze WSL. Inside that range memory is flat on JAX ≥0.11.
- **Adjoint memory isn't flat on JAX 0.10,** the oldest supported version: about 0.4 states per layer, against
  backprop's 12. The test checks a weaker bound there, and the docs and the changelog say so.
- **On the GPU an adjoint gradient holds about 12.5 states at 18 qubits,** against about 5 on the CPU. It doesn't grow
  with depth, but it is more memory than the plan's "about three states" assumed for capacity on an 8 GB GPU. The
  27- and 26-qubit adjoint limits in plan.md §4.5 are unverified, and Sprint 7 should measure them.
- **There is no automatic GPU check.** A JAX, CUDA or driver update can break the GPU without anyone seeing it until
  the next manual run.

**Bugs found:** none reached `main` or a release. CI caught two problems in this sprint's own PRs:

- **The adjoint memory test failed on JAX 0.10** (#151), which led to the documented limit above.
- **PennyLane 0.44's `default.qubit` can't apply a broadcast `GlobalPhase`** (#152). It's PennyLane's limit, in the
  test's reference only: the reference now runs one tape per batch entry.

### Flow metrics (plan.md §4.2)

| Metric | Value |
|---|---|
| Throughput | 17 of 17 points (15 committed and the 2-point stretch), 6 of 6 stories. QS-907's 3 points were dropped, not carried (Sprint 0: 16, Sprint 1: 18, Sprint 2: 21, Sprint 3: 23, Sprint 4: 28, Sprint 5: 23) |
| Cycle time | Median 8 h 49 min from opening a story's PR to merging it (8 minutes to 5 d 23 h). #153 and #155 sat for almost 6 days, because a dropped macOS job or an out-of-date branch stopped their auto-merge while nobody was watching |
| Sprint duration | 2026-11-17 to 2026-11-24 |
| WIP limits | **Not held at the end:** #152, #153, #154 and #155 were open together, which is what made them queue behind each other |
| Escaped defects | 0 |
| CI failures before merge | 2 caused by code or tests (above). **GitHub never assigned a macOS machine** on #151, #152, #153 and twice on `main`, and each needed a manual rerun |
| ADR amendments | ADR-0007 D2 (two corrections) and D5 (no runner), approved by the PO |

### Demo

A batch of training runs in one compiled step, with adjoint gradients, on the RTX 4060:

```python
energy = expval_fn(ansatz, [H], diff_method="adjoint")
step = jax.jit(jax.vmap(lambda x: x - 0.5 * jax.grad(lambda v: energy(v)[0])(x)))
```

```text
device {CudaDevice(id=0)} batch (64, 1) qubits 4
first step 0.42 s
steps 41, 0.14 s after compile; start energies -1.1349 to 0.4797
FCI -1.1372697289; final energies -1.1372697289 to -1.1372697289; worst error 2.7e-12; reached 64/64
```

## Retrospective

| Keep | Change | Try |
|---|---|---|
| **Measure before deciding.** Spikes corrected the plan's claim about `jax.hessian`, found the barrier the adjoint needed, and put real noise numbers behind ADR-0008 | **PRs stalled for six days.** Auto-merge stops when GitHub can't find a macOS machine or a branch falls behind, and nothing reran or updated them | Retry only jobs GitHub never started, automatically (a PO decision, because it changes CI) |
| **Test both ends of every dependency range.** The minimum-versions job caught both of this sprint's CI failures | **Too many PRs open at once.** Required up-to-date branches make them queue, each with a full CI run | Hold to one story PR in review at a time |
| **Run on the real GPU before claiming it works.** The GPU run showed a fixed scratch area and a higher state count that the CPU couldn't | **The app doesn't wake the session on a CI failure unless Auto-fix is on.** The PO was told it would, and it didn't | Say which PRs are being watched, and turn on Auto-fix only when the PO asks |

## Forecast update

Throughput over seven sprints: 16, 18, 21, 23, 28, 23 and 17 points (mean 20.9). 172 points remain across Sprints
7–16, after this sprint's 17 and QS-907's 3 dropped points. That's about 8.2 sprints at the current pace, against the
10 planned. Next: release **v0.2.0**, then Sprint 7's benchmarks under ADR-0008.
