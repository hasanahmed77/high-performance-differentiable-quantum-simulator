# Getting started

## Requirements

- Python 3.12–3.15
- Linux, macOS or Windows

## Install

```bash
pip install ketjet                 # the simulator; needs only NumPy
pip install "ketjet[pennylane]"    # plus the PennyLane device, see "PennyLane device"
pip install "ketjet[jax]"          # plus the JAX backend, see "Backends"
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

## Backends

A backend holds the state and runs the gates. Every circuit method takes one:

| Backend | Install | What it's for |
|---|---|---|
| `NumPyBackend()` (default) | `pip install ketjet` | The reference implementation: plain NumPy, the simplest to read, and the one every other backend is tested against. It also has adjoint differentiation with memory independent of depth. |
| `JaxBackend()` | `pip install "ketjet[jax]"` | The same results on JAX arrays. Its functions can be traced by JAX, which is what compilation, backprop and GPU execution build on. |

```python
import jax
from ketjet.backends import get_backend

jax.config.update("jax_enable_x64", True)   # needed for complex128, see below

backend = get_backend("jax")                 # or get_backend("numpy")
bell.probs(backend=backend)                  # array([0.5, 0. , 0. , 0.5])
bell.expval(Z(0) @ Z(1), backend=backend)    # 1.0
```

Both backends return the same NumPy types, and one seed gives the same samples on both.

**JAX and precision.** JAX computes in 32 bits unless its 64-bit mode is on, and would quietly return complex64
when asked for complex128. ketjet raises an error instead, and never changes JAX's settings for you. Either enable
64-bit mode once at the start of your program, as above, or pass `dtype="complex64"`.

**Which JAX versions.** ketjet's JAX backend supports JAX 0.10 and later. PennyLane 0.45's own JAX interface works
only with JAX below 0.11, so install `"jax<0.11"` if you use JAX through PennyLane.

**Compiling a circuit.** `compile_circuit` turns a circuit into a `jax.jit`-compiled function of one flat array of
parameter values. JAX compiles it once per circuit *structure* (its wires, gates and controls), so a loop that runs
one circuit shape with new values, as every optimizer does, compiles only on the first call:

```python
from ketjet.backends.jax_backend import compile_circuit, parameter_values

run = compile_circuit(circuit)               # nothing is compiled yet
state = run(parameter_values(circuit))       # the first call compiles
state = run(new_values)                      # later calls reuse the program
```

Slot `k` of the array is the circuit's `k`-th gate parameter, in operation order, and a symbol that appears in two
gates fills two slots. A bound circuit and its symbolic template share one compilation, and the most recent 128
structures stay compiled. Compiling a circuit of several hundred gates takes seconds, so a circuit that runs only
once is faster on NumPy.

**What's not there yet.** `jax.grad` through a compiled circuit, batching and GPU execution are the next steps on
the [roadmap](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator#roadmap). Its
`adjoint_jacobian` returns the same values as NumPy's but uses reverse-mode autodiff, so its memory grows with
circuit depth.

## Next steps

- [API reference](reference.md)
- [Design decisions](adr/0001-statevector-representation.md)
