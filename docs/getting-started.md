# Getting started

## Requirements

- Python 3.12–3.15
- Linux, macOS or Windows

## Install

```bash
pip install ketjet                 # the simulator; needs only NumPy
pip install "ketjet[pennylane]"    # plus the PennyLane device, see "PennyLane device"
```

The latest development version installs from source:

```bash
pip install git+https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator.git
```

## A Bell state in five lines

```python
import ketjet as kj
from ketjet.observables import X, Z

bell = kj.Circuit(2).h(0).cnot(0, 1)

bell.probs()                  # array([0.5, 0. , 0. , 0.5])
bell.expval(Z(0) @ Z(1))      # 1.0: the two qubits always agree
bell.var(X(0))                # 1.0
```

## Building circuits

`Circuit(n)` is a fluent builder: every gate method appends an operation and returns the circuit.

```python
c = (
    kj.Circuit(3)
    .h(0)
    .cnot(0, 1)
    .crx(0.25, 1, 2)       # controlled rotation: angle first, then control and target
    .toffoli(0, 1, 2)
)
print(c)    # Circuit(3) [H(0), CNOT(0, 1), CRX(0.25, 1, 2), Toffoli(0, 1, 2)]
```

Wire 0 is the most significant bit everywhere (the same convention as PennyLane), so `probs()` of
`|01⟩` puts probability 1 at index 1.

## Parameters

Build a circuit once with symbolic parameters, then bind values for each run:

```python
from ketjet.operations import Parameter

theta = Parameter("theta")
ansatz = kj.Circuit(2).ry(theta, 0).cnot(0, 1)

for value in (0.0, 0.5, 1.0):
    print(ansatz.bind(theta=value).expval(Z(0)))   # cos(theta): 1.0, 0.878, 0.540
```

## Observables and Hamiltonians

Observables are written the way they're written on paper:

```python
from ketjet.observables import Hermitian, X, Y, Z

H = 0.5 * Z(0) @ Z(1) - 0.2 * X(0) + 1.3 * Y(1)
bell.expval(H)
```

Pauli strings are applied without building a matrix, so Hamiltonians with many terms stay cheap.
`Hermitian(matrix, wires)` covers any other observable.

## Sampling and shots

Real hardware returns samples, not probabilities. ketjet can do the same, reproducibly:

```python
bell.sample(5, seed=7)                       # (5, 2) array of correlated bits
bell.expval(X(0), shots=10_000, seed=7)      # ≈ 0.0, with statistical noise
```

## Precision

Everything runs in `complex128` by default. Use `dtype="complex64"` to halve memory:

```python
bell.probs(dtype="complex64").dtype   # float32
```

## Next steps

- [API reference](reference.md)
- [Design decisions](adr/0001-statevector-representation.md)
