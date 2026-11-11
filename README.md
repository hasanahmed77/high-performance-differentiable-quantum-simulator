# ketjet

**A high-performance differentiable quantum statevector simulator, usable as a PennyLane device.**

[![PyPI](https://img.shields.io/pypi/v/ketjet)](https://pypi.org/project/ketjet/)
[![CI](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/ci.yml)
[![Docs](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/actions/workflows/docs.yml/badge.svg?branch=main)](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/LICENSE)
[![Type checked: mypy strict](https://img.shields.io/badge/types-mypy%20strict-informational)](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/pyproject.toml)

> **Status: alpha (v0.1.0).** The NumPy reference simulator, with exact gradients, as a PennyLane device. The
> faster backends (JAX, C++, CUDA) come next; see the [roadmap](#roadmap) and the
> [project board](https://github.com/users/hasanahmed77/projects/2).

## Why

Quantum machine-learning research runs mostly on classical simulators. Training a parametrized circuit means
simulating it thousands of times and computing gradients each time. Researchers need a simulator that is:

- **differentiable**, with exact, memory-efficient gradients for deep circuits;
- **fast** on CPUs and GPUs, and **batched**, so many parameter sets run as one computation;
- **trustworthy**: verified against established tools, with honest performance numbers.

## Quickstart

```bash
pip install "ketjet[pennylane]"
```

Use it from PennyLane like any other device, here with ketjet's adjoint gradients:

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("ketjet.qubit", wires=2)

@qml.qnode(dev, diff_method="adjoint")
def energy(theta):
    qml.RY(theta[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(theta[1], wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1))

theta = np.array([0.3, 0.8], requires_grad=True)
print(f"{energy(theta):.6f}")             # 0.844467
print(qml.grad(energy)(theta).round(6))   # [ 0.477668 -0.717356]

opt = qml.GradientDescentOptimizer(stepsize=0.2)
for _ in range(100):
    theta = opt.step(energy, theta)
print(f"{energy(theta):.6f}")             # -1.500000, the exact minimum
```

Or use ketjet's own API, which needs nothing but NumPy:

```python
import ketjet as kj
from ketjet import gradients
from ketjet.observables import X, Z
from ketjet.operations import Parameter

a, b = Parameter("a"), Parameter("b")
circuit = kj.Circuit(2).ry(a, 0).cnot(0, 1).rx(b, 1)
H = Z(0) @ Z(1) + 0.5 * X(0) @ X(1)
values = {"a": 0.3, "b": 0.8}

print(f"{circuit.bind(values).expval(H):.6f}")                                # 0.844467
print(gradients.grad(circuit, H, values).round(6))                            # [ 0.477668 -0.717356]
print(gradients.grad(circuit, H, values, method="parameter-shift").round(6))  # [ 0.477668 -0.717356]
```

More runnable examples (a Bell state, the QFT and a VQE for H₂), each checked in CI, are in
[`examples/`](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/tree/main/examples).

## What you get

- **A PennyLane device**, `ketjet.qubit`: every PennyLane gate and template runs, by decomposition into ketjet's
  gates. `expval`, `var` and `probs` are native; shots are seeded and reproducible.
  [Device docs](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/pennylane/)
- **Exact gradients three ways:**
  - **adjoint** (the default), with memory independent of circuit depth;
  - **parameter-shift**, with exact rules derived from each gate's spectrum, including the four-term rule for
    controlled rotations;
  - **finite differences**, as a test oracle.
- **Vector-Jacobian products in one backward pass**, however many expectation values a circuit returns: the form
  PennyLane and JAX ask for when training.
- **A small, typed Python API:** a fluent circuit builder, symbolic parameters, Pauli and Hermitian observables,
  and both precisions (complex128 and complex64).

## Results

What is verified today, on every pull request across Linux, macOS and Windows and Python 3.12–3.15:

| Claim | Evidence |
|---|---|
| **Correct** | States, probabilities, expectation values, variances and Jacobians match PennyLane's `default.qubit` to 1e-10 (complex128) on random circuits |
| **Conformant** | PennyLane's own device suite passes: 380 tests in exact mode and 365 with shots, **0 failures**; every skip is [explained](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/pennylane/#conformance) |
| **Adjoint memory is independent of depth** | Peak memory is 5.1 to 5.35 statevectors from 20 to 4000 layers (14 qubits); storing every intermediate state would need one per gate |
| **Gradients are exact** | Adjoint, parameter-shift and PennyLane's backprop agree on random circuits, including through gates PennyLane decomposes |
| **One backward pass per VJP** | A VJP over 4 expectation values makes 1 backend call (checked by counting) |
| **Specialized kernels** | Diagonal and permutation gates run 2 to 10× faster than ketjet's dense kernel from 16 qubits up (Apple M2) |
| **A real problem** | A VQE on the H₂ molecule [reaches the exact ground-state energy](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/examples/vqe-h2/) to 1e-12 |
| **Tested** | 100% line and branch coverage today (CI requires at least 85%); 80 deliberately injected bugs caught, several only after a test was strengthened |

**Not yet measured: speed against other simulators.** This release is the NumPy *reference* backend, built to be
correct and readable, so that every faster backend can be checked against it. Benchmarks against `default.qubit`,
`lightning.qubit`, Qiskit Aer and Qulacs will be published with the faster backends, with reproducible scripts and
plots, **including the cases where ketjet loses**.

## Design

ketjet is layered so that every faster backend is checked against a simple, readable reference:

```
PennyLane device (ketjet.qubit) · Python API (Circuit, gradients)
        │  circuit IR: an inspectable list of operations
Differentiation ── adjoint · parameter-shift · finite differences
        │  one typed backend protocol
NumPy (reference, today) · JAX (jit, vmap, GPU) · C++ (SIMD, OpenMP) · CUDA
```

Every significant decision is recorded with its alternatives in an architecture decision record:
[statevectors](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/adr/0001-statevector-representation/),
[circuit IR](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/adr/0002-circuit-ir/),
[measurements](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/adr/0003-measurements-and-observables/),
[differentiation](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/adr/0004-differentiation/)
and the
[PennyLane device](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/adr/0005-pennylane-device/).

## Installation

```bash
pip install ketjet                 # the simulator; needs only NumPy
pip install "ketjet[pennylane]"    # plus the PennyLane device
```

Requires Python 3.12–3.15 on Linux, macOS or Windows. To install the latest development version from source:

```bash
pip install "ketjet[pennylane] @ git+https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator.git"
```

## Roadmap

| Release | Adds |
|---|---|
| **v0.1.0** (released) | NumPy reference simulator, exact gradients, PennyLane device |
| v0.2.0 | JAX backend: jit, vmap batching, GPU, backprop and adjoint inside JAX; then the first benchmarks |
| v0.3.0 | Native C++ backend: SIMD and OpenMP, with prebuilt wheels |
| v0.4.0 | Gate fusion and autotuning, with a CPU roofline and a performance report |
| v0.5.0 | CUDA backend, with GPU adjoint |
| v1.0.0 | A stable API, complete docs and a technical write-up |

## Documentation

[hasanahmed77.github.io/high-performance-differentiable-quantum-simulator](https://hasanahmed77.github.io/high-performance-differentiable-quantum-simulator/):
getting started, the PennyLane device, the H₂ example, the API reference, design decisions and sprint reports.

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/CONTRIBUTING.md)
for setup, checks and workflow, and
[SECURITY.md](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/SECURITY.md)
for reporting vulnerabilities.

## License

[Apache License 2.0](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator/blob/main/LICENSE)
