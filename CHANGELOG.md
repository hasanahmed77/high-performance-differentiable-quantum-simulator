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
