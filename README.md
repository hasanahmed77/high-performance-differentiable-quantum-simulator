# ketjet

**A high-performance differentiable quantum statevector simulator, usable as a PennyLane device.**

[![CI](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/ci.yml)
[![Docs](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/docs.yml/badge.svg?branch=main)](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Type checked: mypy strict](https://img.shields.io/badge/types-mypy%20strict-informational)](pyproject.toml)

> **Status: pre-release (Sprint 0, Foundations).** The first usable release, v0.1.0, is planned for the end of Sprint 4.
> Follow progress on the [project board](https://github.com/users/hasanahmed77/projects/2).

## The problem

Quantum machine-learning research runs mostly on classical simulators. Training a parameterized circuit
means simulating it thousands of times and computing gradients each time. Researchers need a simulator that is:

- **fast** on CPUs and GPUs,
- **differentiable**, with memory-efficient gradients for deep circuits,
- **batched**, so many circuits or parameter sets run as one computation,
- **trustworthy**: verified against established tools, with honest performance numbers.

## Design

ketjet is layered so that every faster backend is checked against a simple, readable reference:

```
Python API / PennyLane device
        │  circuit IR (inspectable list of operations)
Compiler passes ── decomposition, gate fusion
        │
Differentiation ── adjoint · parameter-shift · backprop (JAX)
        │  one typed backend protocol
NumPy (reference) · JAX (jit/vmap/GPU) · C++ (SIMD/OpenMP) · CUDA
```

Full architecture, trade-offs and decision records: see [`plan.md`](plan.md) (the design doc follows in later sprints).

## Results

Benchmarks against `default.qubit`, `lightning.qubit`/`lightning.gpu`, Qiskit Aer and Qulacs will be
published here from Sprint 7, with reproducible scripts and plots, **including the cases where ketjet loses**.

## Installation

Not on PyPI yet (planned for v0.1.0: `pip install ketjet`). To install from source:

```bash
pip install git+https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator.git
```

Requires Python 3.12–3.15 on Linux, macOS or Windows.

## Documentation

[hasanahmed77.github.io/high-performance-differentiable-quantum-simulator](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/)

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks and workflow, and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[Apache License 2.0](LICENSE)
