# Project Plan: High-Performance Differentiable Quantum Simulator

> A fast statevector simulator built for speed, batching and automatic differentiation,
> with NumPy, JAX, native C++ and CUDA backends, shipped on PyPI as a PennyLane device.

| | |
|---|---|
| **Repository** | `hasanahmed77/high-performance-differentiable-quantum-simulator` |
| **Framework** | Scrum with goal-based sprints + Kanban flow (GitHub Projects) |
| **Product Owner / Architect** | Project owner: sets priorities, approves designs, reviews every PR, accepts or rejects sprint output |
| **Development team** | Claude: implements, tests, documents, proposes designs as ADRs |
| **Status** | Planning (Sprint 0 not yet started) |

---

## Table of contents

1. [Product vision and goals](#1-product-vision-and-goals)
2. [The quality bar (non-negotiable)](#2-the-quality-bar-non-negotiable)
3. [Architecture overview](#3-architecture-overview)
4. [Way of working (Agile setup)](#4-way-of-working-agile-setup)
5. [Definitions of Ready and Done](#5-definitions-of-ready-and-done)
6. [Testing strategy](#6-testing-strategy)
7. [Product backlog](#7-product-backlog)
8. [Sprint plan](#8-sprint-plan)
9. [Release plan](#9-release-plan)
10. [Quality bar traceability](#10-quality-bar-traceability)
11. [Risks and mitigations](#11-risks-and-mitigations)
12. [Open decisions for the Product Owner](#12-open-decisions-for-the-product-owner)

---

## 1. Product vision and goals

### Vision

For quantum-ML researchers and PennyLane users who need fast simulation with gradients, this
project is a statevector simulator that is **correct first, then fast, then honest about its speed**.
Unlike a teaching simulator, it is engineered like production software: layered backends, a
compiler pass for gate fusion, a choice of memory-efficient differentiation methods, native
kernels, and benchmarks that are published along with the cases where it loses.

### Goals (measurable)

| # | Goal | Target |
|---|---|---|
| G1 | Correctness | Matches PennyLane `default.qubit` to 1e-10 (complex128) or 1e-5 (complex64) on ≥10,000 random circuits; passes PennyLane's `pl-device-test` suite |
| G2 | Differentiation | Backprop (JAX), adjoint and parameter-shift; adjoint memory stays O(1) state copies regardless of depth |
| G3 | Performance (CPU) | Faster than `default.qubit` on all suite benchmarks at ≥20 qubits; within 1.5× of `lightning.qubit` (published either way) |
| G4 | Performance (GPU) | ≥70% of peak memory bandwidth on single-gate kernels (roofline-verified) |
| G5 | Scale | 29 qubits (complex64) on the 8 GB development GPU; 30 qubits on a 16 GB T4 (free tier); 31 qubits sharded across 2× T4 (stretch) |
| G6 | Distribution | `pip install ketjet` works on Linux/macOS/Windows for Python 3.12–3.15, with prebuilt wheels |
| G7 | Adoption | ≥1 external user or contributor; ≥1 upstream PR to the PennyLane ecosystem |

### Constraint: zero budget

The project runs at **$0 total cost**. No paid services, cloud rentals, paid CI minutes or
subscriptions. Every tool and service below is free for public open-source repositories.

| Need | Free solution |
|---|---|
| CPU CI (Linux, macOS, Windows) | GitHub Actions standard runners (free and unlimited for public repos) |
| GPU development and GPU CI | Owned RTX 4060 workstation (self-hosted runner) |
| Large-scale GPU runs | Kaggle Notebooks (2× T4, 16 GB each, weekly free quota) and Google Colab free tier (1× T4) |
| Coverage gate and reporting | pytest-cov + diff-cover inside CI (no account); Codecov optional for badges |
| Docs and benchmark dashboard hosting | GitHub Pages |
| Package distribution | PyPI / TestPyPI |
| Security scanning | Dependabot, CodeQL, pip-audit |
| Project management | GitHub Projects |
| Profiling | Nsight Compute / Nsight Systems, `perf`, py-spy (all free) |

Any proposal that introduces a cost is rejected at refinement.

### Explicitly out of scope

Noise models and density matrices, tensor-network or MPS simulation, real hardware execution,
pulse-level control, and a custom circuit language. These are listed so the scope stays fixed.

---

## 2. The quality bar (non-negotiable)

Every one of these must be true before v1.0.0, and most from v0.1.0 onward.
Section 10 maps each item to the backlog stories that deliver it.

| QB | Requirement |
|---|---|
| QB1 | Clear README: problem, design, results, how to install |
| QB2 | Released on PyPI with versioned releases and a changelog |
| QB3 | Type hints + linting (ruff, mypy) enforced in CI |
| QB4 | CI matrix (multiple Python versions, OSes) |
| QB5 | 85%+ meaningful test coverage, plus regression tests |
| QB6 | Docs site (Sphinx or MkDocs) with API reference and examples |
| QB7 | Benchmarks with reproducible scripts and plotted results |
| QB8 | A short design doc explaining the architecture and trade-offs |
| QB9 | A write-up (blog post or report) linkable from a CV |
| QB10 | At least one real user or outside contributor (stretch) |

**Enforcement rule:** QB3, QB4 and QB5 are automated CI gates from Sprint 0. A PR that lowers
coverage below 85%, fails mypy `--strict`, or fails on any matrix cell cannot be merged.

---

## 3. Architecture overview

The full rationale lives in `docs/design.md` and in the ADRs (`docs/adr/NNNN-*.md`). This section
is the map.

```
 ┌───────────────────────────────────────────────────────────────┐
 │  User-facing layer                                            │
 │   • Python API: Circuit, gates, observables, run()/expval()   │
 │   • PennyLane device adapter (qml.device("ketjet.qubit"))      │
 └───────────────┬───────────────────────────────────────────────┘
                 │  Circuit IR (immutable list of Operations)
 ┌───────────────▼───────────────────────────────────────────────┐
 │  Compiler layer (passes over the IR)                          │
 │   • validation → decomposition → gate fusion → scheduling     │
 └───────────────┬───────────────────────────────────────────────┘
                 │  Optimized IR
 ┌───────────────▼───────────────────────────────────────────────┐
 │  Differentiation layer                                        │
 │   • backprop (via JAX) • adjoint • parameter-shift • finite-diff│
 └───────────────┬───────────────────────────────────────────────┘
                 │  Backend protocol (apply_op, expval, sample, ...)
 ┌───────┬───────▼──────┬──────────────────┬────────────────────┐
 │ NumPy │ JAX          │ Native CPU (C++) │ CUDA (C++/CUDA)    │
 │ (ref) │ jit/vmap/GPU │ SIMD + OpenMP    │ single + multi-GPU │
 └───────┴──────────────┴──────────────────┴────────────────────┘
```

**Key architectural principles**

1. **The NumPy backend is the oracle.** It is simple and readable, and every faster backend is tested against it.
2. **The IR is data.** Circuits are inspectable lists of operations, so compiler passes such as fusion can rewrite them.
3. **One backend protocol.** Every backend implements the same typed `Protocol`, and the same test suite runs on all of them (parametrized fixtures).
4. **Specialized kernels over dense matrices.** Diagonal gates (Z, RZ, CZ, phase) and permutation gates (X, CNOT, SWAP) get fast paths. Fusion must not destroy them.
5. **Precision is explicit.** complex64 and complex128 are both supported, and tolerances are defined per precision.
6. **Qubit ordering matches PennyLane** (wire 0 = most significant bit), so integration needs no reindexing.

---

## 4. Way of working (Agile setup)

### 4.1 Roles

| Role | Who | Responsibilities |
|---|---|---|
| Product Owner + Architect | Project owner | Owns and orders the backlog, approves ADRs, reviews PRs, runs sprint review, accepts or rejects increments |
| Development team | Claude | Refines stories, estimates, implements, tests, documents, writes ADRs and sprint reports |

### 4.2 Cadence

- **Goal-based sprints (no fixed timebox).** The team works at maximum daily pace. A sprint is
  scoped by its **sprint goal** and closes as soon as that goal is met and the sprint-level DoD (§5.3)
  passes, not on a calendar date. Review and retro still close every sprint, so speed never bypasses
  the quality gates.
- **Scope is fixed once a sprint starts.** New work goes to the backlog, not into the running sprint.
  If a sprint can't reach its goal, the PO either re-scopes it explicitly or ends it early and re-plans.
- **Sprint 0** is a foundations sprint that delivers tooling, not features.

**Flow metrics** (since there's no fixed timebox, these replace calendar velocity), recorded in each sprint report:

| Metric | Definition | Use |
|---|---|---|
| Throughput | Points (and stories) completed per sprint | Forecasting the remaining roadmap |
| Cycle time | In Progress → Done, per story | Spotting stories that are too big or get blocked |
| Review wait | In Review → merged | Detecting a PO-review bottleneck |
| Sprint duration | Planning → review, in days | Trend over time; forecast for the next releases |
| Escaped defects | Bugs found after a story reached Done | Quality of the DoD |

| Ceremony | When | Output |
|---|---|---|
| Backlog refinement | Ongoing; before each planning | Stories meet the Definition of Ready, are estimated and split to ≤8 points |
| Sprint planning | Day 1 | Sprint goal + selected stories moved Backlog → Ready, approved by PO |
| Daily sync | Async via the board | Board kept current; blockers flagged with `blocked` label |
| Sprint review | Last day | Demo, test report, coverage, benchmark deltas, acceptance by PO |
| Retrospective | After review | `docs/retros/sprint-NN.md`: keep / change / try |

### 4.3 GitHub Projects board

**Project:** "Quantum Simulator Roadmap" (Projects v2, linked to the repo).

**Status columns (Kanban flow):**

| Column | Entry rule | WIP limit |
|---|---|---|
| 📥 Backlog | Any issue | none |
| ✅ Ready | Meets the Definition of Ready, selected for the current sprint | none |
| 🔨 In Progress | Branch created, work started | **2** |
| 👀 In Review | PR open, CI green, awaiting PO review | **3** |
| 🧪 Verify | Merged; sprint-level integration/regression checks pending | none |
| 🚀 Done | Meets the Definition of Done | none |

**Sprints as milestones:** each sprint is a GitHub **Milestone** (`Sprint 00 – Foundations`, …) with no due date,
closed at sprint review. Milestones suit goal-based sprints better than fixed-length Iteration fields.

**Custom fields:** `Sprint` (milestone), `Epic` (single-select E1–E12), `Priority` (P0 Must / P1 Should / P2 Could / P3 Won't-now),
`Points` (number, Fibonacci 1-2-3-5-8), `Type` (feature, bug, test, docs, chore, spike, perf).

**View:** a single *Sprint Board* (board layout, columns by Status). The Product backlog and the Roadmap are
read from the issue list and the milestones (each milestone groups one sprint's stories and shows its progress),
so extra project views aren't needed. *(PO decision, 2026-11-09.)*

**Status transitions and WIP limits** are maintained by the development team through the GitHub API as part of
the workflow, rather than by built-in project automations:
- story started → 🔨 In Progress; PR opened → 👀 In Review; PR merged → 🧪 Verify; sprint review → 🚀 Done.
- The WIP limits (In Progress ≤2, In Review ≤3) are checked before a story starts or a PR opens, and recorded in
  the sprint report if they are exceeded.

**Labels:** `epic:*`, `type:*`, `P0`–`P3`, `blocked`, `good first issue`, `help wanted`, `breaking-change`, `perf-regression`.

**Issue templates:** User story (story, acceptance criteria, test plan, QB items touched), Bug, Spike (question, timebox, expected output), ADR proposal.

### 4.4 Engineering workflow

- **Branching:** trunk-based. `main` is protected, and work happens on short-lived branches named `<type>/QS-<id>-<slug>` (e.g. `feat/QS-021-apply-gate`).
- **Commits:** Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`, `chore:`, `refactor:`).
- **PRs:** one story per PR, linked to the issue (`Closes #N`), with CI green. Squash merge.
- **Merge policy (PO decision, 2026-11-10):** the development team merges a PR **once every required check is green**.
  GitHub doesn't let a PR author approve their own PR (all commits come from the PO's account), so branch protection
  requires a PR and green status checks (0 required approvals). The PO reviews the increment at **sprint review**,
  can comment on or request changes to any PR at any time, and **ADRs still need explicit PO approval before
  implementation**.
- **Versioning:** SemVer. Pre-1.0 minors may break the API, and every break is listed in the changelog.
- **Changelog:** Keep-a-Changelog format, generated from towncrier news fragments, one per PR.
- **ADR numbering:** ADRs are numbered in the order they're written (`docs/adr/NNNN-*.md`); backlog stories
  don't reserve numbers.
- **Design changes:** any change to a public API, the backend protocol, or the IR needs an ADR approved by the PO *before* implementation.

### 4.5 Development environments and compute

| Environment | Hardware | OS / toolchain | Role |
|---|---|---|---|
| **Primary dev** | MacBook Air M2 (ARM64, Apple GPU not used) | macOS, Python 3.12–3.15, Clang | Day-to-day development of all CPU work; NEON SIMD path; macOS arm64 wheel validation |
| **GPU workstation** | Desktop PC, NVIDIA RTX 4060 (8 GB, Ada, compute capability 8.9), x86-64 | Windows 11 + **WSL2 Ubuntu 24.04**, CUDA 12.x toolkit, `jax[cuda12]`, GCC/Clang, Nsight Compute | JAX-GPU and CUDA development, GPU benchmarks and roofline, x86 AVX2 validation, native Windows wheel testing |
| **Hosted CI** | GitHub-hosted runners | Ubuntu, macOS, Windows | Full CPU matrix on every PR (§6.4) |
| **Free GPU tier** | Kaggle Notebooks: 2× NVIDIA T4 (16 GB each); Google Colab free: 1× T4 (backup) | Hosted Linux notebooks; dependencies installed from the pinned lockfile | Large-scale runs only: 30-qubit single-GPU target (G5) and real 1 → 2 GPU scaling (E10). Planned, time-boxed sessions within the free weekly quota. |

**Why WSL2:** JAX's CUDA builds, `lightning.gpu` and cuStateVec are Linux-only. WSL2 provides
a full Linux CUDA stack on the Windows workstation without dual-booting. Native Windows remains a
supported *target platform* (CPU wheels), tested in hosted CI.

**Single-GPU capacity (RTX 4060, 8 GB):**

| Workload | complex64 | complex128 |
|---|---|---|
| Statevector simulation | ≤29 qubits | ≤28 qubits |
| Adjoint differentiation (~3 state copies) | ≤27 qubits | ≤26 qubits |

**Free-tier capacity (T4, 16 GB):** ≤30 qubits complex64 on one T4; ≤31 qubits complex64 sharded across Kaggle's 2× T4.

**Environment reproducibility:** a pinned `environment/gpu.Dockerfile`, a pinned lockfile and a setup guide
(`docs/dev/gpu-workstation.md`) define the GPU environment. The workstation runs the Docker image.
Kaggle and Colab can't run custom images, so they install the same pinned lockfile from a versioned
notebook (`benchmarks/notebooks/`). Every benchmark records its hardware, driver and package versions, so
results from different machines are comparable.

**GPU CI (self-hosted runner) security model.** The repository is public, so the runner is hardened:
- It runs inside WSL2 under a dedicated non-admin user, labelled `self-hosted, linux, gpu`.
- `gpu.yml` triggers **only** on `workflow_dispatch`, `schedule` and `push` to `main`, and never on `pull_request`. Code from forks never reaches the workstation.
- Repository setting: *require approval for all outside collaborators' workflows*.
- Jobs run in an ephemeral container (the image above) with no repository secrets mounted.
- The workstation is not always on. Scheduled jobs queue until the runner comes online, and the sprint-level DoD for GPU sprints requires a green `gpu.yml` run before sprint review.

**When each environment is needed:**

| Sprint | Mac | GPU workstation | Free GPU tier (Kaggle/Colab) |
|---|---|---|---|
| 0–5 | ● | ○ (setup at end of Sprint 5) | |
| 6 | ○ | ● JAX-GPU, runner live | |
| 7 | ● | ● GPU baselines | |
| 8 | ● | | |
| 9 | ● | ○ AVX2 / Windows wheel validation | |
| 10–11 | ● | ○ GPU fusion sweep (QS-807) | |
| 12–13 | ○ | ● CUDA backend, GPU roofline | ● one session: 30-qubit run (QS-908) |
| 14 | ● (multi-device emulated on CPU) | ○ | ● Kaggle 2× T4: scaling runs |
| 15–16 | ● | ○ final GPU regression run | |

● primary, ○ occasional

---

## 5. Definitions of Ready and Done

### 5.1 Definition of Ready (story can enter a sprint)

- [ ] Written as a user story or a clearly scoped technical task
- [ ] Acceptance criteria are testable (Given/When/Then or a checklist)
- [ ] Test plan names the test types required (unit / integration / property / regression / perf)
- [ ] Estimated at ≤8 points (larger stories are split)
- [ ] Dependencies are done or scheduled earlier in the same sprint
- [ ] ADR approved if the story touches a public API, the protocol or the IR

### 5.2 Definition of Done: story level

- [ ] Code is fully type-hinted; `mypy --strict` and `ruff check` / `ruff format --check` pass
- [ ] Unit tests written; integration tests where the story crosses a component boundary
- [ ] Coverage of changed lines ≥90%; project coverage stays ≥85%
- [ ] Any bug fix includes a regression test that fails before the fix
- [ ] CI is green on the full matrix
- [ ] Public API has docstrings (NumPy style) and appears in the API reference
- [ ] Changelog fragment added
- [ ] PO has reviewed and approved the PR

### 5.3 Definition of Done: sprint level

- [ ] Every story in Done meets the story-level DoD
- [ ] Full test suite runs on `main`: unit, integration, property-based, regression, and the PennyLane device suite (once it exists)
- [ ] Benchmark suite runs on `main` with no unexplained regression above 5%
- [ ] Docs site builds without warnings and is deployed
- [ ] Sprint report written to `docs/sprints/sprint-NN.md`: goal met?, stories done/carried, test results, coverage, benchmark deltas, flow metrics (§4.2)
- [ ] Retrospective recorded

### 5.4 Definition of Done: release level

- [ ] All sprint-level DoD items met
- [ ] Release candidate published to **TestPyPI**, then installed and smoke-tested in clean venvs on all three operating systems
- [ ] Changelog finalized; version tagged (`vX.Y.Z`); GitHub Release notes published
- [ ] Published to PyPI through Trusted Publishing (OIDC, no stored tokens)
- [ ] README results section and benchmark plots updated
- [ ] Design doc updated for any architectural change

---

## 6. Testing strategy

### 6.1 Test types

| Type | Tool | What it proves | Runs |
|---|---|---|---|
| **Unit** | pytest | Each function or kernel is correct in isolation (gate matrices unitary, known-state outputs) | Every PR |
| **Property-based** | Hypothesis | Invariants hold for arbitrary inputs: norm preserved, U·U† = I, gate then inverse = identity, measurement probabilities sum to 1 | Every PR (fast profile); nightly (deep profile) |
| **Differential** | pytest + PennyLane | Random circuits give the same states, expvals and gradients as `default.qubit` and as our NumPy oracle | Every PR (500 circuits); nightly (10,000) |
| **Cross-backend** | parametrized fixtures | Every backend (NumPy, JAX, native, CUDA) matches the NumPy oracle | Every PR (CPU); GPU workflow (GPU) |
| **Gradient checks** | pytest | Adjoint = backprop = parameter-shift ≈ finite difference | Every PR |
| **Integration** | pytest | Components work together: IR → passes → backend → measurement; PennyLane QNode end-to-end; JAX jit/vmap/grad composition | Every PR |
| **Device conformance** | `pl-device-test` | Passes PennyLane's official device test suite | Every PR, from Sprint 4 |
| **Regression** | pytest + golden files | Bugs never return; golden outputs for canonical circuits (QFT, GHZ, VQE ansatz) are stable | Every PR |
| **Performance regression** | asv (airspeed velocity) | No silent slowdowns; >5% regression fails the nightly run and opens a `perf-regression` issue | Nightly |
| **Memory** | tracemalloc / `jax` memory stats | Adjoint uses O(1) state copies; no leaks in the native backend | Nightly |
| **Native code safety** | ASan/UBSan builds, C++ unit tests (GoogleTest/Catch2) | No out-of-bounds or undefined behaviour in kernels | Every PR touching `native/` |
| **Packaging smoke** | clean venv install of built wheel | The wheel installs and imports; the device registers with PennyLane | Every release and nightly |
| **Docs tests** | doctest / `mkdocs build --strict` | Every example in the docs actually runs | Every PR |

### 6.2 Test organization

```
tests/
  unit/           # fast, isolated
  property/       # Hypothesis
  integration/    # cross-component, PennyLane, JAX transforms
  differential/   # vs default.qubit and the NumPy oracle
  regression/     # one file per fixed bug + golden outputs
  native/         # C++ tests (built by CMake/ctest)
benchmarks/       # asv suite + comparison scripts
```

Pytest markers: `slow`, `gpu`, `native`, `pennylane`, `nightly`. Default PR runs exclude `slow`, `gpu` and `nightly`.

### 6.3 Numerical tolerance policy

| Precision | Absolute tolerance | Relative tolerance |
|---|---|---|
| complex128 | 1e-10 | 1e-8 |
| complex64 | 1e-5 | 1e-4 |
| Gradients vs finite difference | 1e-5 (step 1e-7, complex128) | n/a |

Tolerances live in one module (`tests/tolerances.py`) and are never loosened inline without an ADR.

### 6.4 CI pipelines (GitHub Actions)

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | PR, push to main | lint (ruff), types (mypy), tests on {ubuntu, macos, windows} × {3.12, 3.13, 3.14, 3.15}, coverage gates (pytest-cov project ≥85%, diff-cover patch ≥90%), build + wheel smoke test, docs build |
| `native.yml` | PR touching `native/` | CMake build, C++ tests, sanitizers, wheel build test |
| `nightly.yml` | cron, daily | deep Hypothesis, 10k differential circuits, asv benchmarks, memory tests, packaging smoke |
| `gpu.yml` | manual, weekly, push to `main` (never PRs) | JAX-GPU + CUDA tests and GPU benchmarks on the self-hosted RTX 4060 runner (§4.5) |
| `release.yml` | tag `v*` | cibuildwheel wheels + sdist → TestPyPI → smoke → PyPI (Trusted Publishing) → GitHub Release |
| `docs.yml` | push to main | Build and deploy the docs site |
| Security | scheduled | Dependabot, CodeQL, `pip-audit` |

---

## 7. Product backlog

**ID scheme:** `QS-<epic><nn>`. **Priority:** P0 = Must (needed for the next release), P1 = Should, P2 = Could, P3 = Won't-now.
**Points:** Fibonacci, relative effort. The backlog is ordered top-down within each epic, and the PO reorders freely.

### E1: Engineering foundation and developer experience

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-101 | As a maintainer, I want a `src/` layout package with `pyproject.toml` (hatchling), so it builds and installs cleanly | P0 | 2 | `pip install -e .[dev]` works; `python -m build` produces sdist + wheel |
| QS-102 | As a maintainer, I want ruff + mypy `--strict` + pre-commit hooks, so style and types never drift | P0 | 2 | Pre-commit runs ruff, ruff-format, mypy; config in `pyproject.toml` |
| QS-103 | As a maintainer, I want a CI matrix across 3 OSes × 4 Python versions, so the code works everywhere | P0 | 3 | `ci.yml` green on 12 cells; required status checks on `main` |
| QS-104 | As a maintainer, I want a coverage gate at 85%, so tests stay meaningful | P0 | 1 | `pytest --cov-fail-under=85` (project) and `diff-cover --fail-under=90` (patch) in CI; PR fails below |
| QS-105 | As the PO, I want the GitHub Projects board, fields, views, labels and templates set up, so the process is visible | P0 | 3 | Everything in §4.3 exists; backlog imported as issues |
| QS-106 | As a maintainer, I want branch protection and a PR template, so every change is reviewed | P0 | 1 | Protected `main`: PR required, required status checks, linear history, no force-push; PO merge is the approval (§4.4); PR checklist includes the DoD |
| QS-107 | As a contributor, I want CONTRIBUTING, CODE_OF_CONDUCT, SECURITY and LICENSE files, so I know how to participate | P0 | 2 | Files present; license per decision D3 |
| QS-108 | As a maintainer, I want towncrier changelog fragments enforced in CI, so the changelog is never forgotten | P1 | 2 | PR without a fragment fails unless labelled `skip-changelog` |
| QS-109 | As the PO, I want ADR and sprint-report templates, so decisions and progress are recorded | P0 | 1 | `docs/adr/0000-template.md`, `docs/sprints/template.md` |
| QS-110 | As a maintainer, I want Dependabot, CodeQL and pip-audit, so dependencies stay safe | P1 | 2 | Workflows active, no high-severity alerts |
| QS-111 | Spike: choose the PyPI package name and prepare its first release (decision D1) | P0 | 1 | Name chosen; trusted publishers registered on PyPI and TestPyPI; release pipeline merged; **no empty placeholder upload**, since PEP 541 treats a package with no functionality as name squatting. The first PyPI upload is the first release with real functionality. |
| QS-112 | As a maintainer, I want a reproducible GPU workstation environment (WSL2 Ubuntu, CUDA 12, pinned Docker image, setup guide) | P0 | 3 | `nvidia-smi` and `jax.devices()` show the RTX 4060 inside the container; guide in `docs/dev/gpu-workstation.md` |

### E2: Core simulation engine (NumPy reference backend)

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-201 | ADR-0001: statevector representation, qubit ordering and precision policy | P0 | 2 | ADR approved: `(2,)*n` tensor view, wire 0 = MSB, complex64/128 |
| QS-202 | As a user, I want to allocate an n-qubit `\|0…0⟩` state in either precision | P0 | 1 | Correct shape and dtype; memory estimate helper raises a clear error if too large |
| QS-203 | As a developer, I want a generic `apply_matrix(state, U, wires)` kernel for k-qubit gates | P0 | 5 | Correct for any wire order; property tests: norm preserved, U then U† = identity |
| QS-204 | As a user, I want the standard gate library (I, X, Y, Z, H, S, S†, T, T†, SX, RX, RY, RZ, Phase, Rot, U3, CNOT, CZ, CY, SWAP, iSWAP, CRX/CRY/CRZ, Toffoli, CSWAP, multi-controlled) | P0 | 5 | Every matrix verified unitary and against PennyLane's matrices |
| QS-205 | As a developer, I want specialized kernels for diagonal and permutation gates | P1 | 5 | Same results as the dense kernel; benchmarked speedup recorded |
| QS-206 | As a user, I want controlled versions of any gate (`ctrl(op, controls)`) | P1 | 3 | Matches the dense controlled matrix |
| QS-207 | As a user, I want arbitrary unitary gates (`QubitUnitary`) with an optional unitarity check | P1 | 2 | Non-unitary input raises unless the check is disabled |
| QS-208 | As a user, I want state preparation (basis state, amplitude vector) | P1 | 3 | Normalization validated; matches PennyLane |
| QS-209 | As a developer, I want a typed `Backend` protocol, so new backends plug in identically | P0 | 3 | `typing.Protocol`; NumPy backend conforms; mypy enforces |

### E3: Circuit API, IR and measurements

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-301 | ADR-0002: Circuit IR design (immutable op list, parameters, wires) | P0 | 2 | Approved; supports later fusion and parameter tracking |
| QS-302 | As a user, I want a fluent `Circuit` builder (`c.h(0).cnot(0, 1)`) | P0 | 3 | Builds the IR; `repr`, equality, serialization round-trip |
| QS-303 | As a user, I want symbolic or trainable parameters, so circuits can be re-run with new values | P0 | 3 | Parameter binding; trainable-parameter indices tracked |
| QS-304 | As a user, I want `state()`, `probs(wires)`, `expval(obs)`, `var(obs)` | P0 | 5 | Match PennyLane on random circuits |
| QS-305 | As a user, I want observables: Pauli strings, Hermitian matrices, tensor products, Hamiltonians (weighted sums) | P0 | 5 | Pauli expvals use fast paths (no dense matrices); match PennyLane |
| QS-306 | As a user, I want `sample(shots, seed)` and shot-based expvals, so results are reproducible | P0 | 3 | Same seed gives same samples; statistical test (chi-square) passes |
| QS-307 | As a user, I want circuit drawing in text | P2 | 2 | Snapshot tests |
| QS-308 | As a user, I want mid-circuit measurement and reset | P3 | 5 | Deferred; tracked for after v1 |

### E4: Differentiation

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-401 | As a researcher, I want finite-difference gradients as a test reference | P0 | 2 | Central difference; used only in tests and as a fallback |
| QS-402 | As a researcher, I want parameter-shift gradients, so gradients are exact and hardware-compatible | P0 | 3 | Matches finite difference; supports multi-term shift rules for controlled rotations |
| QS-403 | ADR: adjoint differentiation design (memory, supported ops, observables) | P0 | 2 | Approved |
| QS-404 | As a researcher, I want adjoint differentiation for expvals, so deep circuits differentiate with O(1) extra memory | P0 | 8 | Matches param-shift; memory test proves constant state copies vs depth |
| QS-405 | As a researcher, I want a Jacobian for multiple observables in one adjoint pass | P1 | 3 | One backward pass per circuit, not per observable |
| QS-406 | As a researcher, I want vector-Jacobian products (VJP), so the device composes with any autodiff framework | P0 | 3 | VJP matches J^T·v |
| QS-407 | As a researcher, I want gradients through Hamiltonian expvals (VQE-style) | P1 | 2 | VQE H2 example converges to the known energy |

### E5: PennyLane integration

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-501 | Spike: study PennyLane's `devices.Device` API, preprocessing and the test suite | P0 | 2 | Notes in an ADR for the PennyLane device |
| QS-502 | As a PennyLane user, I want `qml.device("ketjet.qubit", wires=n)` via an entry point | P0 | 5 | Device registered through `pennylane.plugins`; QNodes execute |
| QS-503 | As a PennyLane user, I want unsupported ops decomposed automatically | P0 | 3 | Preprocess transforms; unsupported ops give a clear error |
| QS-504 | As a PennyLane user, I want `diff_method="adjoint"` and `"backprop"` to work | P0 | 5 | `compute_derivatives` / `compute_vjp` implemented; matches `default.qubit` |
| QS-505 | As a maintainer, I want `pl-device-test` passing in CI | P0 | 3 | All applicable tests pass; skips documented |
| QS-506 | As a PennyLane user, I want parameter broadcasting (batched parameters) | P1 | 3 | Matches `default.qubit` broadcasting results |
| QS-507 | As a maintainer, I want compatibility tested against the latest and minimum supported PennyLane versions | P1 | 2 | CI job pins min version |

### E6: JAX backend (jit, vmap, autodiff, GPU)

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-601 | ADR: JAX backend design (pure functions, static wires, dtype handling) | P0 | 2 | Approved |
| QS-602 | As a user, I want the JAX backend to pass the full cross-backend suite | P0 | 5 | All oracle tests pass |
| QS-603 | As a user, I want circuits to be `jax.jit`-compiled, so repeated runs are fast | P0 | 3 | No retracing for the same structure (trace-count test) |
| QS-604 | As a researcher, I want `jax.grad` through the simulator (backprop) | P0 | 3 | Matches adjoint and param-shift |
| QS-605 | As a researcher, I want `jax.vmap` batching over parameters, so a batch of circuits runs as one kernel | P0 | 5 | Batched results equal a loop; benchmark shows speedup |
| QS-606 | As a researcher, I want adjoint exposed to JAX via `jax.custom_vjp`, so I get adjoint memory inside JAX programs | P1 | 5 | `jax.grad` uses adjoint when selected; memory test |
| QS-607 | As a user, I want the JAX backend on GPU | P0 | 3 | GPU workflow green; results match CPU within tolerance |
| QS-608 | As a user, I want PennyLane's JAX interface to work end-to-end with our device | P0 | 3 | QNode with `interface="jax"` jit/grad works |

### E7: Native CPU backend (C++)

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-701 | ADR: native core (C++20, nanobind, scikit-build-core, memory ownership, threading) | P0 | 3 | Approved |
| QS-702 | As a maintainer, I want the CMake + scikit-build-core + nanobind build working on all OSes | P0 | 5 | Editable install builds the extension; CI builds on 3 OSes |
| QS-703 | As a user, I want native 1- and 2-qubit gate kernels with zero-copy NumPy interop | P0 | 5 | Cross-backend suite passes; no copies (buffer-pointer test) |
| QS-704 | As a user, I want OpenMP multithreading with a configurable thread count | P0 | 3 | Scales on benchmarks; deterministic results |
| QS-705 | As a user, I want SIMD-vectorized kernels (AVX2/AVX-512 on x86, NEON on ARM) with runtime dispatch | P1 | 8 | Same results as scalar; speedup recorded |
| QS-706 | As a user, I want native specialized kernels (diagonal, permutation, controlled) | P1 | 5 | Cross-backend tests; benchmarks |
| QS-707 | As a researcher, I want native adjoint differentiation | P0 | 5 | Matches the NumPy adjoint; memory test |
| QS-708 | As a maintainer, I want C++ unit tests and ASan/UBSan in CI | P0 | 3 | `native.yml` green with sanitizers |
| QS-709 | As a user, I want prebuilt wheels (cibuildwheel) for Linux x86_64/aarch64, macOS arm64/x86_64 and Windows | P0 | 5 | Wheels install in clean venvs with no compiler |

### E8: Circuit optimization (gate fusion)

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-801 | ADR: fusion strategy (greedy, max-k qubits, which gates are excluded, parameter handling) | P0 | 2 | Approved |
| QS-802 | As a developer, I want a compiler-pass framework over the IR | P0 | 3 | Passes are composable, typed and tested in isolation |
| QS-803 | As a user, I want greedy gate fusion with a configurable max fused size k | P0 | 5 | Fused circuit equals the original (property test on random circuits) |
| QS-804 | As a user, I want fusion to keep diagonal/permutation fast paths, not turn them into dense matrices | P0 | 3 | Policy tests; benchmarks show no regression |
| QS-805 | As a researcher, I want fusion to keep gradients correct (fuse only non-trainable runs, or track parameters) | P0 | 5 | Gradients identical with fusion on/off |
| QS-806 | As a user, I want an autotuner that picks k per machine and caches it | P1 | 3 | Picks the best k within 5% on the benchmark suite |
| QS-807 | As the PO, I want a fusion-size sweep (k = 1…7, CPU and GPU) with plots | P0 | 2 | Plot in the docs + write-up |

### E9: CUDA backend

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-901 | ADR: CUDA backend (kernel design, memory management, packaging strategy) | P0 | 2 | Approved |
| QS-902 | As a user, I want CUDA kernels for k-qubit dense, diagonal and permutation gates | P0 | 8 | Cross-backend suite passes on GPU |
| QS-903 | As a user, I want GPU expvals and sampling without copying the state to the host | P0 | 5 | Only scalars/samples cross PCIe (profiler-verified) |
| QS-904 | As a researcher, I want GPU adjoint differentiation | P1 | 5 | Matches CPU adjoint |
| QS-905 | As a user, I want complex64 and complex128 on GPU | P0 | 2 | Both pass with their tolerances |
| QS-906 | As a user, I want an installable GPU package (separate CUDA wheel extra, e.g. `pip install ketjet[cuda12]`) | P1 | 5 | Installs on a clean CUDA 12 machine |
| QS-907 | As a maintainer, I want a hardened self-hosted GPU CI runner on the workstation | P0 | 3 | Meets the §4.5 security model; `gpu.yml` never triggers on PRs; green run on `main` |
| QS-908 | As the PO, I want a 30-qubit run on a free-tier T4 (Kaggle or Colab) to validate G5 | P1 | 2 | Versioned notebook installs the pinned lockfile; results and hardware recorded in the benchmark JSON; $0 cost |

### E10: Multi-GPU and distributed (stretch)

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-1001 | ADR: state partitioning (global vs local qubits, qubit-swap strategy) | P2 | 3 | Approved |
| QS-1002 | As a user, I want the statevector sharded across GPUs with `jax.sharding` | P2 | 8 | Matches single-device results; 1 more qubit per doubling of devices |
| QS-1003 | As a developer, I want a communication-minimizing scheduler (reorder or swap qubits to reduce cross-device gates) | P2 | 8 | Fewer collectives than naive, measured |
| QS-1004 | As the PO, I want measured 1 → 2 GPU scaling (Kaggle 2× T4) plus a communication-cost analysis from emulated 1/2/4/8-device runs | P2 | 3 | Plots in docs + write-up; measured and emulated results clearly labelled |

### E11: Benchmarking and performance analysis

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-1101 | ADR: benchmark methodology (warmup, repeats, reported statistics, hardware disclosure) | P0 | 2 | Approved |
| QS-1102 | As the PO, I want a benchmark suite: single-gate throughput, QFT, random circuits, a VQE ansatz with gradients, batched QML | P0 | 5 | Scripts reproducible with one command; JSON results committed |
| QS-1103 | As the PO, I want comparisons vs `default.qubit`, `lightning.qubit`, `lightning.gpu`, Qiskit Aer and Qulacs | P0 | 5 | Same circuits, same hardware; losses reported |
| QS-1104 | As the PO, I want auto-generated plots (time vs qubits, speedup, gradient cost) | P0 | 3 | `make plots` regenerates every figure |
| QS-1105 | As a maintainer, I want asv performance-regression tracking, nightly | P1 | 3 | Dashboard published; >5% regression opens an issue |
| QS-1106 | As the PO, I want a roofline analysis (CPU + GPU) of the kernels | P1 | 5 | Measured bandwidth vs peak (STREAM / Nsight Compute); plot + explanation |
| QS-1107 | As the PO, I want a profiling and optimization report: memory, fusion, batching, threading | P0 | 3 | `docs/performance.md` with before/after numbers |

### E12: Docs, release and community

| ID | Story | Pri | Pts | Acceptance criteria |
|---|---|---|---|---|
| QS-1201 | As a user, I want a docs site with getting started, user guide and API reference | P0 | 5 | Deployed; `--strict` build; API docs auto-generated |
| QS-1202 | As a user, I want runnable example notebooks/scripts (Bell, QFT, VQE, QML classifier, batching) | P0 | 3 | Executed in CI |
| QS-1203 | As a visitor, I want a README with problem, design, results, install, quickstart and badges | P0 | 2 | Updated at every release |
| QS-1204 | As the PO, I want `docs/design.md`, a short architecture and trade-offs doc | P0 | 3 | ≤6 pages; links to ADRs; reviewed by the PO |
| QS-1205 | As a maintainer, I want an automated release pipeline (TestPyPI → smoke → PyPI, Trusted Publishing) | P0 | 5 | Dry run + real release succeed |
| QS-1206 | As the PO, I want a public technical write-up (blog/report) | P0 | 5 | Published; linked from the README and CV |
| QS-1207 | As the PO, I want an API stability and deprecation policy for v1 | P1 | 2 | Documented; deprecation warnings tested |
| QS-1208 | As the PO, I want an upstream contribution to PennyLane or Lightning | P1 | 5 | PR opened (merge is a bonus) |
| QS-1209 | As the PO, I want outreach to get a first external user or contributor (PennyLane forum, discussion, `good first issue`s) | P2 | 3 | ≥5 good first issues labelled; announcement posted |

**Backlog totals:** 95 stories across 12 epics. Every story except QS-308 (deferred, P3) is scheduled in §8. Each sprint's selection is confirmed at sprint planning.

---

## 8. Sprint plan

Every sprint ends with the **sprint-level DoD (§5.3)**: the full test suite on `main`, a benchmark run,
docs deployed, a sprint report and a retro. Test focus lists the test types that are *new* in that sprint.

### Sprint 0: Foundations
**Goal:** An empty but production-grade repo where every quality gate is already enforced.
**Stories:** QS-101, 102, 103, 104, 105, 106, 107, 109, 111, 1201 (skeleton only), 1203 (skeleton only)
**Test focus:** CI pipeline itself; a placeholder test proves the matrix and coverage gate work.
**Demo:** A PR that breaks typing is blocked, and a clean PR merges.

### Sprint 1: The engine
**Goal:** Apply any gate to any qubits correctly.
**Stories:** QS-201, 202, 203, 204, 209, 301
**Test focus:** Unit tests (every gate matrix); property tests (unitarity, norm, inverse); first differential tests vs PennyLane matrices.
**Demo:** Bell and GHZ states built from raw kernels.

### Sprint 2: Circuits and measurements
**Goal:** Users can build circuits and get every measurement type out.
**Stories:** QS-302, 303, 304, 305, 306, 108
**Test focus:** Differential testing (500 random circuits vs `default.qubit`); statistical tests for sampling; regression golden files (QFT, GHZ).
**Demo:** `Circuit` API notebook.

### Sprint 3: Differentiation
**Goal:** Exact gradients three ways, with adjoint memory proven.
**Stories:** QS-401, 402, 403, 404, 406, 205
**Test focus:** Gradient checks (adjoint = param-shift ≈ finite diff); memory tests; integration test for an optimization loop.
**Demo:** A VQE-style optimization on H2.

### Sprint 4: PennyLane device and **v0.1.0**
**Goal:** Installable from PyPI and usable as a PennyLane device.
**Stories:** QS-501, 502, 503, 504, 505, 1202 (first examples), 1205, 1203
**Test focus:** `pl-device-test` conformance; integration tests for QNodes; packaging smoke tests on 3 OSes.
**Release:** **v0.1.0**, the NumPy reference simulator with adjoint, as a PennyLane plugin.

### Sprint 5: JAX backend I
**Goal:** JAX backend at parity with NumPy, jit-compiled.
**Stories:** QS-601, 602, 603, 604, 608, 110, 112
**Test focus:** Cross-backend parametrized suite; trace-count tests (no recompiles); JAX interface integration tests.
**Demo:** jit'd circuit vs NumPy timing.
**Environment:** Mac; GPU workstation set up at sprint end (QS-112) so Sprint 6 starts unblocked.

### Sprint 6: JAX backend II (batching, GPU) and **v0.2.0**
**Goal:** Batched and GPU execution with gradients.
**Stories:** QS-605, 606, 607, 506, 907 (GPU runner), 405
**Test focus:** vmap-equals-loop tests; GPU-vs-CPU tolerance tests; adjoint-in-JAX memory tests.
**Release:** **v0.2.0**, the JAX backend with jit, vmap, grad and GPU.
**Environment:** 🖥️ GPU workstation required.

### Sprint 7: Benchmarks and first performance report
**Goal:** Honest, reproducible numbers against the competition.
**Stories:** QS-1101, 1102, 1103, 1104, 1105, 407
**Test focus:** Performance-regression tracking switched on; benchmark scripts tested so they can't rot.
**Demo:** First comparison plots, including where we lose.
**Environment:** Mac for CPU benchmarks; 🖥️ GPU workstation for `lightning.gpu` and JAX-GPU baselines.

### Sprint 8: Native core I
**Goal:** A C++ extension that builds everywhere and is correct.
**Stories:** QS-701, 702, 703, 704, 708
**Test focus:** C++ unit tests; sanitizers; cross-backend suite on native; zero-copy tests.
**Demo:** Native vs NumPy vs JAX on CPU.

### Sprint 9: Native core II and **v0.3.0**
**Goal:** Fast native CPU backend with adjoint, shipped as wheels.
**Stories:** QS-705, 706, 707, 709, 507
**Test focus:** SIMD-vs-scalar equivalence; native adjoint gradient checks; wheel smoke tests on every platform.
**Release:** **v0.3.0**, the native CPU backend with prebuilt wheels.
**Environment:** Mac (NEON); 🖥️ workstation briefly for AVX2 benchmarks and native Windows wheel checks.

### Sprint 10: Gate fusion
**Goal:** Fusion that is provably result-preserving and gradient-preserving.
**Stories:** QS-801, 802, 803, 804, 805
**Test focus:** Property tests (fused ≡ unfused on random circuits); gradient equality with fusion on/off; regression tests on canonical circuits.
**Demo:** The same circuit before and after fusion, with the speedup.

### Sprint 11: Tuning, roofline and **v0.4.0**
**Goal:** Know where the speed limit is, and get close to it on CPU.
**Stories:** QS-806, 807, 1106 (CPU part), 1107, 1204 (v1 draft), 206, 207
**Test focus:** Autotuner tests; performance-regression baselines reset with justification.
**Release:** **v0.4.0**, gate fusion and autotuning, plus the first performance report.

### Sprint 12: CUDA backend I
**Goal:** Correct CUDA kernels for every gate class.
**Stories:** QS-901, 902, 905, 903
**Test focus:** Cross-backend suite on GPU; host-device transfer checks with the profiler.
**Demo:** CUDA vs JAX-GPU vs Lightning-GPU.
**Environment:** 🖥️ GPU workstation required.

### Sprint 13: CUDA backend II and **v0.5.0**
**Goal:** GPU adjoint, packaging and the GPU roofline.
**Stories:** QS-904, 906, 908, 1106 (GPU part), 208
**Test focus:** GPU gradient checks; GPU wheel install smoke test.
**Release:** **v0.5.0**, the CUDA backend.
**Environment:** 🖥️ GPU workstation required; one free-tier T4 session for QS-908.

### Sprint 14: Multi-GPU (stretch)
**Goal:** Simulate past a single GPU's memory.
**Stories:** QS-1001, 1002, 1003, 1004
**Test focus:** Sharded-vs-single-device equivalence; scaling benchmarks.
**Environment:** Mac for development (JAX emulates 8 devices on the CPU, so correctness work needs no GPU); Kaggle's free 2× T4 for the measured scaling runs.
**Note:** If the PO drops the stretch goal, this sprint becomes hardening.

### Sprint 15: Hardening and v1.0.0-rc
**Goal:** API frozen, docs complete, every quality-bar item green.
**Stories:** QS-1204 (final), 1207, 1201 (complete), 307, bug bash
**Test focus:** Full nightly suite run on demand; coverage audit (meaningful, not padded); docs examples executed; security scan clean.
**Release:** **v1.0.0rc1** to TestPyPI and PyPI.

### Sprint 16: v1.0.0 and launch
**Goal:** Ship and tell people about it.
**Stories:** QS-1206, 1208, 1209, final README
**Test focus:** Release-level DoD (§5.4).
**Release:** **v1.0.0**, plus the write-up, the upstream PR and outreach.

**Total:** 17 goal-based sprints. There's no calendar estimate up front: after Sprint 2, the measured throughput
and sprint durations (§4.2) are used to forecast the remaining releases, and the forecast is refreshed at every sprint review.

---

## 9. Release plan

| Version | Sprint | Contents | Quality bar items first satisfied |
|---|---|---|---|
| v0.1.0 | 4 | NumPy backend, Circuit API, measurements, adjoint + param-shift, PennyLane device | QB1, QB2, QB3, QB4, QB5, QB6 (early) |
| v0.2.0 | 6 | JAX backend: jit, vmap, grad, GPU, custom_vjp adjoint | |
| v0.3.0 | 9 | Native C++ CPU backend, SIMD + OpenMP, prebuilt wheels | QB7 (first benchmarks in Sprint 7) |
| v0.4.0 | 11 | Gate fusion, autotuning, CPU roofline, performance report | QB8 (design doc v1) |
| v0.5.0 | 13 | CUDA backend, GPU adjoint, GPU roofline | |
| v1.0.0 | 16 | Stable API, complete docs, write-up, upstream PR | QB9, QB10 |

---

## 10. Quality bar traceability

| QB | Delivered by | First met | Kept true by |
|---|---|---|---|
| QB1 README | QS-1203 | Sprint 4 | Release DoD: updated every release |
| QB2 PyPI + changelog | QS-111, 108, 1205 | Sprint 4 | Automated release pipeline |
| QB3 Types + lint in CI | QS-102, 103 | Sprint 0 | Required CI check |
| QB4 CI matrix | QS-103, 709 | Sprint 0 | Required CI check |
| QB5 85% coverage + regression tests | QS-104, story DoD | Sprint 0 | CI coverage gates; regression test required per bug fix |
| QB6 Docs site | QS-1201, 1202 | Sprint 4 (complete by 15) | `docs.yml` + strict build |
| QB7 Benchmarks + plots | QS-1101–1107 | Sprint 7 | Nightly asv; plots regenerated per release |
| QB8 Design doc | QS-1204 + ADRs | Sprint 11 (final in 15) | ADR required for architectural changes |
| QB9 Write-up | QS-1206 | Sprint 16 | n/a |
| QB10 Outside user/contributor | QS-1208, 1209 | Sprint 16 (stretch) | Good first issues kept stocked |

---

## 11. Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Development GPU has 8 GB, which limits single-GPU scale | Medium | Certain | 8 GB covers all development and roofline work (§4.5); 30-qubit and 2-GPU results come from free-tier T4 sessions using the pinned lockfile |
| `ketjet` name taken before our first real release | Low | Low | Obscure, distinctive name; pending trusted publishers registered; first real release published as early as policy allows (no placeholder uploads, per PEP 541) |
| Free GPU tiers change their quotas or availability | Low | Medium | Free-tier results are nice-to-have, never release-blocking; Colab is the fallback for Kaggle; the 4060 covers everything else |
| Zero budget limits multi-GPU to 2 devices | Low | Certain | 1 → 2 GPU measurements prove the distributed design and its communication cost; larger scaling is analysed with emulated devices and labelled as such |
| GPU work is split across two machines (Mac + workstation) | Medium | Medium | Pinned container image; GPU code behind the backend protocol so CPU work is never blocked; all state lives in git, so switching machines is a `git pull` |
| Self-hosted runner on a public repo | High | Low | Hardened per §4.5: no PR triggers, outside-collaborator approval, ephemeral container, no secrets |
| WSL2 GPU passthrough or driver issues | Medium | Low | Set up and validated one sprint early (QS-112, Sprint 5); pinned NVIDIA driver and CUDA versions |
| PennyLane device API changes | Medium | Medium | Pin a tested range; CI against latest and minimum; follow PennyLane release notes |
| Can't beat `lightning`/Qulacs on CPU | Medium | High | That's expected: publish honest results; compete on batching + autodiff (our design focus) rather than raw single-circuit CPU speed |
| Cross-platform C++ wheel builds (Windows, OpenMP on macOS) | Medium | High | Sprint 8 is dedicated to the build; cibuildwheel; fall back to non-OpenMP builds on macOS if needed |
| Coverage gaming (85% that tests nothing) | High | Medium | Property-based and differential tests carry the weight; PO reviews test quality, not just the number |
| Scope creep | High | Medium | Out-of-scope list in §1; new ideas go to the backlog at P3 and the PO decides |
| PO review becomes the bottleneck | Medium | Medium | Small PRs (one story each); WIP limit 3 for In Review; a sprint report summarizes everything for review |
| CUDA packaging complexity | Medium | High | Ship the CUDA backend as a separate extra/wheel; source build documented as fallback |
| MkDocs core is unmaintained (no release since Aug 2024); Material for MkDocs's authors now point to their successor, Zensical (0.1.0, Nov 2026) | Low | Medium | Stay on Material (still maintained) for now; Zensical reads `mkdocs.yml`, so migration is low-cost. Re-evaluate at Sprint 15 hardening. |

---

## 12. Open decisions for the Product Owner

To be settled in Sprint 0 planning. The recommendation comes first in each case.

All decisions were settled by the PO on 2026-11-09.

| # | Decision | Outcome | Rationale |
|---|---|---|---|
| D1 | PyPI package name | ✅ **`ketjet`** (`import ketjet as kj`; device `ketjet.qubit`) | Unique and searchable; "ket" is quantum state notation and "jet" signals speed. Available on PyPI as of 2026-11-09, to be reserved in Sprint 0 (QS-111). |
| D2 | Docs tool | ✅ **MkDocs Material + mkdocstrings** | Modern, Markdown-based, fast to iterate |
| D3 | License | ✅ **Apache-2.0** | Matches PennyLane; includes a patent grant |
| D4 | Supported Python | ✅ **3.12–3.15** (revised 2026-11-09 in Sprint 0) | NumPy ≥2.5 and JAX ≥0.12 require Python ≥3.12; follows SPEC 0. PennyLane-dependent tests are skipped on 3.15 until PennyLane supports it. |
| D5 | GPU access | ✅ **RTX 4060 workstation (WSL2) + free-tier Kaggle/Colab T4s** | Zero budget (§1, §4.5) |
| D6 | Sprint length | ✅ **Goal-based sprints, not fixed-length** | Work proceeds at maximum daily pace; a sprint closes when its goal and DoD are met (§4.2) |
| D7 | Multi-GPU (E10) | ✅ **Kept as stretch** | Most impressive piece; scheduled last so it cannot block anything |
| D8 | Build backend | ✅ **hatchling**, then **scikit-build-core** from Sprint 8 | Simple for pure Python; scikit-build-core handles the C++/CUDA build |
