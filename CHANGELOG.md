# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before v1.0.0, minor versions may contain
breaking changes; every break is listed here. From Sprint 2 onward, entries are generated from towncrier
news fragments (one per PR).

## [Unreleased]

### Added
- Package skeleton (`ketjet`), typed (PEP 561), Apache-2.0 licensed.
- Release pipeline: TestPyPI → smoke test → PyPI via Trusted Publishing.
- `ketjet.zero_state(n, dtype, max_bytes=...)`: allocate the |0…0⟩ state in complex128 or complex64, with clear
  `MemoryError`s; `ketjet.state_nbytes` reports memory up front (QS-202).
- `ketjet.kernels.apply_matrix(state, matrix, wires)`: apply any 2^k x 2^k matrix to k wires; pure, dtype-preserving,
  validated against an independent dense reference (QS-203).
- Gate library (QS-204, ADR-0002): `ketjet.gates` registry of 20 base gates with structure flags verified against
  their matrices; `ketjet.operations` with immutable, self-validating `Operation`s, symbolic `Parameter`s, controlled
  gates as structure (`CNOT`, `CZ`, `CRX`, `Toffoli`, `CSWAP`, `MultiControlledX`, `ctrl`), and inverses. Every matrix
  matches PennyLane.
- `ketjet.backends`: typed `Backend` protocol with the consumed-input ownership rule (ADR-0001 D4) and the pure
  `NumPyBackend` reference implementation; random circuits match PennyLane's `default.qubit` (QS-209).

[Unreleased]: https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/commits/main
