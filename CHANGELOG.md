# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before v1.0.0, minor versions may contain
breaking changes; every break is listed here. Entries are generated from towncrier news fragments
(one per PR, see `changes/README.md`).

## [Unreleased]

Unreleased changes live as fragments in [`changes/`](changes/) and are compiled into this file when a version is
released.

<!-- towncrier release notes start -->

## [0.2.0] - 2026-11-24

The JAX release: a JAX backend with compiled circuits, backprop and adjoint gradients inside JAX, `jax.vmap` batching and PennyLane parameter broadcasting, running on NVIDIA GPUs. GPU execution is tested by hand on an RTX 4060, because CI has no GPU.

### Added

- A pinned GPU environment: `environment/gpu.Dockerfile` (Ubuntu 24.04, Python 3.13 and `jax[cuda12]` from `uv.lock`), an environment check `environment/check_gpu.py`, and a setup guide from a clean Windows 11 install through WSL2, Docker and the NVIDIA Container Toolkit (`docs/dev/gpu-workstation.md`). ([#12](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/12))
- PennyLane parameter broadcasting runs as one batched execution on `ketjet.qubit` with `backend="jax"`: an exact broadcast circuit is mapped with `jax.vmap` from one compilation, when executed and under `diff_method="backprop"`, instead of being split into one circuit per value. Results equal `default.qubit`'s. The NumPy backend, shots, adjoint and parameter-shift still split the circuit, as PennyLane's own devices do. ([#42](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/42))
- A JAX backend, `JaxBackend` (`pip install "ketjet[jax]"`), that returns the NumPy backend's results on JAX arrays, in both precisions. Select it with `get_backend("jax")` or `qml.device("ketjet.qubit", backend="jax")`. complex128 needs JAX's 64-bit mode; ketjet raises a clear error instead of silently computing in complex64. ([#45](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/45))
- `compile_circuit` in `ketjet.backends.jax_backend` compiles a circuit with `jax.jit` into a function of one flat array of parameter values. Circuits of one structure share a single compilation, whatever their parameter values and whether they are bound or symbolic; the most recent 128 structures stay compiled. `parameter_values(circuit)` returns a bound circuit's values in the order that function takes. ([#46](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/46))
- Backprop on the JAX backend: `jax.grad`, `jax.jacrev` and `jax.jit` apply to circuits compiled with `compile_circuit`, and the gradients equal the adjoint and parameter-shift results. `parameter_map(circuit)` maps one value per symbol to the compiled slot array, so a shared symbol gets the sum of its slots' gradients. `JaxBackend.adjoint_jacobian` now runs the compiled program, so repeated calls on one circuit structure compile once. ([#47](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/47))
- A compiled circuit can be batched with `jax.vmap`: the batch runs as one program from the same compilation, and combines with `jax.grad` and `jax.jit`. Passing a batch to a compiled circuit directly raises an error that names `jax.vmap`. ([#48](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/48))
- `expval_fn(circuit, observables, diff_method=...)` builds a JAX function of a circuit's symbol values that returns its expectation values. With `diff_method="adjoint"` its gradient is the adjoint method as a `jax.custom_vjp`: memory doesn't grow with circuit depth, and one backward pass serves every observable. `"backprop"` stays the default, because the adjoint method has no forward mode: `jax.jvp` and `jax.jacfwd` raise, while `jax.hessian` works. On JAX 0.10 the memory still grows a little with depth (about 0.4 states per layer, against backprop's 12); JAX 0.11 and later stay flat. ([#49](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/49))
- A GPU test suite, `pytest -m gpu tests/gpu`, runs the JAX backend on an NVIDIA GPU and compares it with the CPU: states, expectation values, backprop and adjoint Jacobians, `jax.vmap` batches, `ketjet.qubit` and its seeded samples, and the adjoint method's memory, which stays flat with circuit depth on the GPU too. Getting started has a new "Running on a GPU" section. ([#50](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/50))
- `diff_method="backprop"` on `ketjet.qubit` with `backend="jax"` and PennyLane's JAX interface: JAX differentiates the simulation itself, under `jax.jit` and to any order, and the circuit runs as a compiled program cached on its structure. It needs exact results; backprop with shots, or with a non-JAX interface, raises a `DeviceError` that names the fix. `"best"` still resolves to adjoint. ([#51](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/51))
- Runnable examples in `examples/`: a Bell state, the quantum Fourier transform and a VQE for H₂, on `ketjet.qubit`; each checks its own result and runs in CI. ([#88](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/88))

### Changed

- The `pennylane` extra now requires PennyLane 0.44.1 or later. PennyLane 0.44.0 can't be imported with current versions of its dependency `diastatic-malt`, so it never worked in a fresh install. CI now tests the lowest supported version of every dependency (NumPy 2.1, PennyLane 0.44.1, JAX 0.10.0, on Python 3.12) and, daily, the latest releases. ([#43](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/43))
- `JaxBackend.adjoint_jacobian` now runs the adjoint method instead of `jax.jacrev`, so its memory no longer grows with circuit depth (ADR-0004 D4). Its results are unchanged. ([#49](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/49))

### Fixed

- complex64 simulations on the JAX backend are accurate to float32 on NVIDIA GPUs. XLA had computed their contractions in TF32 by default, so states and gradients could be off by up to 1e-2.

### Security

- Every pull request now audits all locked dependencies with `pip-audit` and scans the code and workflows with CodeQL, and Dependabot proposes grouped weekly updates of `uv.lock` and the pinned GitHub Actions. See `SECURITY.md`. ([#10](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/10))


## [0.1.0] - 2026-11-11

The first release: the NumPy reference simulator, with exact gradients (adjoint, parameter-shift and finite
differences), as the PennyLane device `ketjet.qubit`. It passes PennyLane's own device test suite.

### Added

- Changelog entries are now written as towncrier fragments in `changes/` and checked in CI. ([#8](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/8))
- `ketjet.zero_state(n, dtype, max_bytes=...)` allocates the |0…0⟩ state in complex128 or complex64 with clear `MemoryError`s; `ketjet.state_nbytes` reports memory up front. ([#14](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/14))
- `ketjet.kernels.apply_matrix(state, matrix, wires)` applies any 2^k × 2^k matrix to k wires: pure, dtype-preserving, and validated against an independent dense reference. ([#15](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/15))
- Gate library: `ketjet.gates` registers 20 base gates with structure flags verified against their matrices; `ketjet.operations` provides immutable, self-validating operations, symbolic parameters, controlled gates (`CNOT`, `CZ`, `CRX`, `Toffoli`, `CSWAP`, `MultiControlledX`, `ctrl`) and inverses. Every matrix matches PennyLane. ([#16](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/16))
- Specialized kernels for diagonal and permutation gates (`kernels.apply_diagonal`, `kernels.apply_permutation`, `kernels.apply_structured`), used by `NumPyBackend` and adjoint differentiation: 2–10× faster than the dense kernel from 16 qubits up. `Operation.is_diagonal` and `Operation.is_permutation` expose the gate's structure. ([#17](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/17))
- `ketjet.backends` adds the typed `Backend` protocol and the `NumPyBackend` reference implementation; random circuits match PennyLane's `default.qubit`. ([#21](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/21))
- `ketjet.Circuit`: a fluent circuit builder (`Circuit(2).h(0).cnot(0, 1)`) with validation, immutable operation snapshots, `inverse()`, readable `repr` and a versioned JSON format. ([#23](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/23))
- `Circuit.parameters`, `Circuit.parameter_slots` and `Circuit.bind(...)` (by `Parameter`, name or keyword) for symbolic, re-bindable circuits; binding returns a new circuit. ([#24](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/24))
- `Circuit.state()`, `Circuit.probs(wires)`, `Circuit.expval(obs)` and `Circuit.var(obs)` run a circuit and return exact results in PennyLane's conventions; the `Backend` protocol gains read-only `probs`, `expval` and `var`; reference implementations live in `ketjet.measurements`. ([#25](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/25))
- Observables in `ketjet.observables`: Pauli strings (`Z(0) @ Z(1)`), `Hermitian` matrices, tensor products and real-weighted sums (`0.5 * Z(0) @ Z(1) - 0.2 * X(0)`). Pauli strings are applied matrix-free in O(2^n). ([#26](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/26))
- `Circuit.sample(shots, wires, seed=...)` and `Circuit.expval(obs, shots=..., seed=...)`: seeded, reproducible sampling (also on the `Backend` protocol) and shot-based expectation values via eigenbasis rotation. ([#27](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/27))
- `ketjet.gradients.grad` / `jacobian` and `Circuit.gradient`: gradients of expectation values with respect to circuit parameters (shared parameters handled by the chain rule); `method="finite-diff"` (central differences) also differentiates variances. ([#30](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/30))
- `method="parameter-shift"`: exact gradients from shift rules derived from each gate's generator spectrum, including the four-term rule for controlled rotations, and shot-based gradients with `shots=`/`seed=`. Gate specs declare `generator_eigvals`. ([#31](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/31))
- `method="adjoint"` (the default): exact adjoint differentiation of expectation values with memory independent of circuit depth, as one `Backend.adjoint_jacobian` call; analytic gate derivatives (`GateSpec.derivative`, `Operation.derivative_matrix`). ([#33](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/33))
- `ketjet.gradients.vjp` computes vector-Jacobian products. With the default adjoint method it is one backward pass, however many observables there are. ([#35](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/35))
- The PennyLane device `ketjet.qubit` (`pip install "ketjet[pennylane]"`, then `qml.device("ketjet.qubit", wires=n)`): PennyLane decomposes circuits into ketjet's gates, which run on a ketjet backend; `expval`, `var` and `probs` are computed by ketjet, with seeded shots, shot vectors and `c_dtype` precision. `import ketjet` doesn't import PennyLane. ([#38](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/38))
- `ketjet.qubit` runs every PennyLane operation that has a decomposition (templates, `QubitUnitary`, `StatePrep`, Ising and excitation gates, …) by decomposing it into ketjet's gates; operations without one raise a `DeviceError` naming the operation. ([#39](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/39))
- `diff_method="adjoint"` on `ketjet.qubit`: device Jacobians from one backend adjoint call per circuit, and vector-Jacobian products (`device_vjp=True`) in one backward pass however many expectation values a circuit returns. `diff_method="best"` picks adjoint for exact expectation values; PennyLane's parameter-shift and finite differences work too, with or without shots. ([#40](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/40))
- `ketjet.qubit` passes PennyLane's own device test suite (`pl-device-test`) in exact and shot mode, checked on every pull request; operations with a matrix but no decomposition (e.g. `DoubleExcitationPlus`, `BlockEncode`) now run too. New docs page: the PennyLane device. ([#41](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/issues/41))
- Package skeleton: typed (PEP 561), Apache-2.0 licensed, tested on Linux, macOS and Windows with Python 3.12–3.15.
- Release pipeline: TestPyPI, an install smoke test on three OSes, then PyPI via Trusted Publishing.
