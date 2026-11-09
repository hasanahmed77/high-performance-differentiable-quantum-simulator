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

[Unreleased]: https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/commits/main
