# ADR-0005: ketjet is a native PennyLane `Device` that translates tapes 1:1

| | |
|---|---|
| **Status** | Accepted (approved by the PO on 2026-11-11 in #126) |
| **Date** | 2026-11-11 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-501 (this spike), QS-502, QS-503, QS-504, QS-505; later QS-506 (broadcasting), QS-507 (version range), QS-606 (JAX backprop) |

## Context

Most users will reach ketjet through PennyLane (`qml.device("ketjet.qubit", wires=n)`), not through its own API. The
device has to:

- run any QNode a user writes, including gates ketjet doesn't have;
- return results PennyLane's own test suite (`pl-device-test`) accepts;
- expose the adjoint gradients and single-pass VJPs from ADR-0004, which are ketjet's main advantage;
- keep `import ketjet` free of PennyLane, a large optional dependency.

The QS-501 spike studied PennyLane 0.45. Its device API is `pennylane.devices.Device`: a device returns a
**preprocessing pipeline** (transforms that rewrite tapes into what it supports), executes tapes, and may provide
derivatives (`compute_derivatives`, `compute_vjp`). PennyLane's built-in `reference.qubit` is a 180-line example.
Plugins register through the `pennylane.plugins` entry-point group. PennyLane 0.45.1 installs on Python 3.12–3.15.

## Decision

### D1. A `Device` subclass, registered by entry point, behind an optional extra

- `ketjet.pennylane.KetjetQubit(Device)` with `name = "ketjet.qubit"`, registered as
  `[project.entry-points."pennylane.plugins"] "ketjet.qubit" = "ketjet.pennylane:KetjetQubit"`.
- PennyLane is an optional extra: `pip install "ketjet[pennylane]"`. `import ketjet` never imports it. A test
  checks this in a subprocess.
- Arguments: `KetjetQubit(wires=None, shots=None, seed=None, c_dtype="complex128")`. `wires=None` takes the tape's
  wires, as `default.qubit` does. `c_dtype` follows `lightning.qubit`'s name for precision.
- Supported range: PennyLane `>=0.44,<0.46`. The upper bound moves up only after CI passes on the new version.
  QS-507 adds a minimum-version CI job. The `python_full_version < '3.15'` marker on the dev dependency is removed
  once CI confirms PennyLane on 3.15.

### D2. Preprocessing does the hard work; translation is strictly 1:1

The device's `preprocess_transforms()` pipeline:

1. `validate_device_wires`;
2. `defer_measurements` (mid-circuit measurements become extra wires; there's no native support yet);
3. `decompose` until every operation is in the **supported set** (D3), with a clear error for anything with no
   decomposition;
4. `validate_observables` and `validate_measurements`;
5. `broadcast_expand`: batched parameters run as a loop until QS-506 adds native broadcasting;
6. with shots only: `split_non_commuting` and `diagonalize_measurements`.

After preprocessing, each tape is **translated** into ketjet `Operation`s. Wire labels map to `0 … n-1` in device
wire order. **Translation never changes a parameter:** each PennyLane operation becomes exactly one ketjet
operation with the same parameters in the same order. Anything that would need a parameter rewritten (for example
`Adjoint(RX(x))`, which is `RX(-x)`) is not in the supported set, so PennyLane's `decompose` rewrites it before the
device sees it. That keeps gradient bookkeeping trivial (D4): tape parameter *k* is exactly one slot.

| Option | Pros | Cons |
|---|---|---|
| **A. Decompose in PennyLane, translate 1:1 (chosen)** | PennyLane owns decomposition rules and their correctness; trainable parameters map to slots directly | Slightly more preprocessing work per tape |
| B. Translate richer ops ourselves (adjoints of rotations, …) | Fewer gates after decomposition | Rewritten parameters break the parameter-to-slot map, and every rule is ours to get right |
| C. Execute PennyLane's matrices (`op.matrix()`) | Supports everything at once | Throws away structure: no fast kernels, no adjoint derivatives, dense matrices on many wires |

### D3. The supported set

- **Gates:** every ketjet base gate under its PennyLane name: `Identity`, `PauliX`/`Y`/`Z`, `Hadamard`, `S`, `T`,
  `SX`, `RX`, `RY`, `RZ`, `PhaseShift`, `Rot`, `U3`, `SWAP`, `ISWAP`.
- **Controlled gates:** `CNOT`, `CY`, `CZ`, `CRX`, `CRY`, `CRZ`, `ControlledPhaseShift`, `Toffoli`, `CSWAP`,
  `MultiControlledX` and `Controlled(base)` for any supported base, with control values. They become ketjet
  operations with controls (ADR-0002 D2). This includes `CRot`, which PennyLane implements as a controlled `Rot`
  with the same three parameters (amended in QS-502; the first draft listed it as decomposed).
- **Adjoints of fixed gates:** `Adjoint(S)`, `Adjoint(T)`, `Adjoint(SX)`, `Adjoint(ISWAP)`, which are ketjet's `Sdg`,
  `Tdg`, `SXdg` and `ISWAPdg` and take no parameters.
- **`GlobalPhase`:** applied by the device as a scalar on the state, so `qml.state()` matches `default.qubit`
  exactly.
- Everything else, including `QubitUnitary`, `StatePrep`, `BasisState` and templates, is decomposed by
  PennyLane into the set above.

### D4. Gradients: adjoint and VJP on the device; everything else through PennyLane

- `supports_derivatives` is true for `gradient_method="adjoint"` on analytic circuits whose measurements are all
  `expval`: the case ADR-0004 supports. For that case, `"best"` resolves to `"adjoint"`.
- `compute_derivatives` and `execute_and_compute_derivatives` call `Backend.adjoint_jacobian` once per tape
  (ADR-0004 D4), with the tape's trainable parameters as slots.
- `compute_vjp` (and `execute_and_compute_vjp`) uses ADR-0004 D6: one backward pass over $\sum_m v_m O_m$ per tape,
  whatever the number of measurements.
- `parameter-shift`, `finite-diff`, `hadamard` and `spsa` are PennyLane's own gradient transforms. The device just
  executes the shifted tapes, so they work with shots too.
- Adjoint with `var`, `probs`, `state` or shots raises PennyLane's usual error, naming `parameter-shift`.
- **`diff_method="backprop"` is not supported on this device.** Backprop needs the simulation itself to be
  traceable by an autodiff framework. ketjet's NumPy kernels are deliberately plain NumPy (the correctness oracle,
  plan.md §3), so autograd, JAX and Torch can't trace them. Backprop arrives with the JAX backend (Sprint 5/6,
  QS-606). **This narrows QS-504's acceptance criteria** from "adjoint and backprop" to "adjoint (device), plus
  PennyLane's parameter-shift and finite differences". The device raises a clear error suggesting adjoint for
  `backprop`.

### D5. Measurements: native where it matters, PennyLane for the rest

| Measurement | Analytic | With shots |
|---|---|---|
| `expval`, `var` | **ketjet** (`Backend.expval`, `var`), observables translated (below) | ketjet's seeded sampler, then PennyLane's `process_samples` |
| `probs` | **ketjet** (`Backend.probs`) | same |
| `sample`, `counts` | n/a | same |
| `state`, `density_matrix`, `purity`, `vn_entropy`, `mutual_info`, … | PennyLane's `process_state` on ketjet's state | PennyLane's rules (most are analytic-only) |

- **Observables:** `PauliX`/`Y`/`Z` and `Identity` (and their products) become the matrix-free `Pauli` path
  (ADR-0003). `Hermitian`, `Prod`, `SProd`, `Sum` and `LinearCombination` translate structurally. Any other
  observable with a matrix becomes `Hermitian` on its wires.
- **Why split it:** `expval`, `var` and `probs` are the hot paths in every optimization loop, and future GPU backends
  (Sprints 6 and 12) must answer them on the device without copying the state to the host. The rarer measurements
  stay correct by construction by reusing PennyLane's definitions.
- **Shots:** the device's `seed` makes a `numpy.random.Generator`. Each tape draws a child seed (as in ADR-0004 D5),
  so a seeded device is reproducible. Shot vectors (partitioned shots) sample the total once and split it into bins,
  as `default.qubit` does.

### D6. Conformance is a CI gate

- `pl-device-test --device=ketjet.qubit` runs in CI on every PR, in exact mode and in stochastic mode (`--analytic=False --shots=10000`). It runs in one
  dedicated job (Ubuntu, Python 3.13) to keep the matrix fast.
- Every skipped test is listed in `docs/pennylane.md` with its reason, for example backprop (D4) or a feature
  outside the supported set.
- **Differential QNode tests** in the main matrix compare `ketjet.qubit` with `default.qubit` on random circuits:
  states, `probs`, `expval`, `var`, and Jacobians for adjoint, parameter-shift and finite differences, in both
  precisions.

## Consequences

- **Positive:**
  - Users get PennyLane's whole gate and template library on day one, through decomposition.
  - Adjoint and VJP make ketjet the cheapest way to differentiate in PennyLane on the CPU for deep circuits,
    given ADR-0004's memory bound.
  - Faster backends (JAX, C++, CUDA) plug in behind the same device without changing it.
- **Negative:**
  - No backprop until the JAX backend. Users who pick it get a clear error, not a slow path.
  - Decomposition can add gates, for example `QubitUnitary` becomes several rotations. Fusion (Sprint 10) recovers the speed.
  - The PennyLane API moves quickly, so the version range is pinned and needs maintenance (QS-507).
- **Follow-ups:**
  - Native broadcasting (QS-506).
  - A minimum-version CI job (QS-507).
  - JAX backprop through `jax.custom_vjp` (QS-606).
  - Native mid-circuit measurements (a new story, if users ask for it).
  - Native `StatePrep` (a new story; today it decomposes).

## Validation

- `pl-device-test` passes, with every skip documented (QS-505).
- Differential QNode tests match `default.qubit` at the standard tolerances (plan.md §6.3).
- **Gradients match `default.qubit` backprop:** for adjoint and VJP, including gates that only reach the device
  through decomposition (for example `qml.adjoint(qml.RX(x))`, `QubitUnitary`, `StronglyEntanglingLayers`). This is the
  test that D2's 1:1 rule holds.
- **One backend call per tape:** a counting test checks that `compute_vjp` with M measurements makes one
  `adjoint_jacobian` call per tape.
- `import ketjet` in a fresh interpreter doesn't load `pennylane`.
