# ketjet

**A high-performance differentiable quantum statevector simulator, usable as a PennyLane device.**

!!! warning "Pre-release"
    ketjet is in its foundations phase. The first usable release, **v0.1.0**, is planned for the end of
    Sprint 4. This site grows with every sprint.

## What it will be

ketjet simulates quantum circuits by evolving a statevector of $2^n$ complex amplitudes. It is designed around
three goals, in order:

1. **Correct first.** Every backend is tested against a simple NumPy reference implementation and against
   PennyLane's `default.qubit`, on thousands of random circuits.
2. **Then fast.** JAX (`jit`, `vmap`, GPU), native C++ kernels with SIMD and OpenMP, CUDA kernels, and gate fusion.
3. **Then honest about its speed.** Reproducible benchmarks against existing simulators, including the cases where
   ketjet loses.

## Planned features

| Area | Features | Release |
|---|---|---|
| Core | Circuit API, gate library, measurements, sampling | v0.1.0 |
| Gradients | Adjoint (O(1) memory), parameter-shift, backprop via JAX | v0.1.0 – v0.2.0 |
| PennyLane | `qml.device("ketjet.qubit", wires=n)` | v0.1.0 |
| JAX | `jit`, `vmap` batching, GPU | v0.2.0 |
| Native | C++ kernels, SIMD, OpenMP, prebuilt wheels | v0.3.0 |
| Optimization | Gate fusion with autotuning | v0.4.0 |
| GPU | CUDA backend | v0.5.0 |

## Project

- Source code: [GitHub](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator)
- Roadmap and process: [`plan.md`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/plan.md)
- License: Apache-2.0
