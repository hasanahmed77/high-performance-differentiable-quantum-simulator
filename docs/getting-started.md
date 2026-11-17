# Getting started

## Requirements

- Python 3.12–3.15
- Linux, macOS or Windows

| Dependency | Supported | Needed for |
|---|---|---|
| NumPy | 2.1 and later | always |
| PennyLane | 0.44.1 – 0.45.x | `ketjet[pennylane]` |
| JAX | 0.10 and later; below 0.11 with PennyLane's JAX interface | `ketjet[jax]` |

CI tests both ends of every range: the lowest versions on Python 3.12, and the latest releases daily.

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

**Gradients by backprop.** A compiled circuit is a pure JAX function, so `jax.grad`, `jax.jacrev` and
`jax.jit` apply to it directly. `parameter_map(circuit)` turns one value per symbol, in `circuit.parameters`
order, into the slot array; a symbol used by several gates gets the sum of their gradients:

```python
import jax
from ketjet.backends.jax_backend import compile_circuit, expectation, parameter_map

run, to_slots = compile_circuit(ansatz), parameter_map(ansatz)

def energy(x):
    return expectation(run(to_slots(x)), H)

step = jax.jit(lambda x: x - 0.5 * jax.grad(energy)(x))   # one compilation for every step
```

The gradients equal `ketjet.gradients`' adjoint and parameter-shift results. Backprop stores the forward pass, so
its memory grows with circuit depth.

**Adjoint gradients in JAX.** `expval_fn` builds the same kind of function from a circuit and its observables, and
`diff_method="adjoint"` differentiates it by the adjoint method: the backward pass undoes one gate at a time, so it
holds a few states however deep the circuit, and one pass serves every observable:

```python
from ketjet.backends.jax_backend import expval_fn

energies = expval_fn(ansatz, [H, Z(0)], diff_method="adjoint")   # f(x) has shape (2,)
step = jax.jit(lambda x: x - 0.5 * jax.grad(lambda v: energies(v)[0])(x))
jacobian = jax.jacrev(energies)(x)                                # shape (2, number of symbols)
```

It works with `jax.grad`, `jax.jacrev`, `jax.hessian`, `jax.jit` and `jax.vmap`. It has no forward mode of its own,
so `jax.jvp` and `jax.jacfwd` raise `TypeError`: use the default, `diff_method="backprop"`, for those. On 12 qubits, XLA's own
accounting of a gradient's temporary memory stays at about 5 states from 4 to 16 layers with the adjoint method,
while backprop's grows by one state per parametrized gate. On JAX 0.10 the adjoint method still grows a little,
by about 0.4 states per layer at 12 qubits, because of that version's compiler; JAX 0.11 and later stay flat.

**Batching with `jax.vmap`.** To run one circuit with many sets of values, map the function with `jax.vmap`.
JAX turns it into one batched program, from the same compilation, and it combines with `jax.grad` and `jax.jit`:

```python
batch = jax.numpy.stack([x0, x1, x2])                      # shape (3, number of symbols)
energies = jax.vmap(energy)(batch)                         # shape (3,)
steps = jax.jit(jax.vmap(lambda x: x - 0.5 * jax.grad(energy)(x)))
```

A compiled circuit takes one flat array, so passing it a batch directly raises an error that points to
`jax.vmap`. A batch holds one state per entry, `batch × 2**n` amplitudes, so it helps most when one circuit is too
small to keep the device busy. On an RTX 4060 in complex64, one batched call against a loop of single calls was
365× faster at 8 qubits (batch 1024), 19× at 12 (256) and 2.2× at 16 (64), and no faster at 20 qubits (16), where
one circuit already fills the GPU.

**What's not there yet.** GPU execution is tested by hand for now; an automatic GPU test run is next on the
[roadmap](https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator#roadmap). PennyLane's
parameter broadcasting runs as one batch on the JAX backend: see [Broadcasting](pennylane.md#broadcasting).

## Next steps

- [API reference](reference.md)
- [Design decisions](adr/0001-statevector-representation.md)
