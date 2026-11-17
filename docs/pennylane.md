# PennyLane device

ketjet ships a PennyLane device, `ketjet.qubit`, so any PennyLane program can run on ketjet. The design is in
[ADR-0005](adr/0005-pennylane-device.md).

```bash
pip install "ketjet[pennylane]"
```

```python
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("ketjet.qubit", wires=2)

@qml.qnode(dev, diff_method="adjoint")
def circuit(x):
    qml.RY(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

x = np.array(0.3, requires_grad=True)
circuit(x)            # cos(0.3)
qml.grad(circuit)(x)  # -sin(0.3)
```

PennyLane is an optional extra: `import ketjet` on its own never imports it.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `wires` | `None` | Number of wires or their labels (any hashable values). `None` uses the circuit's wires. |
| `shots` | `None` | Default shots; prefer `qml.set_shots` on the QNode, as PennyLane recommends. `None` is exact. |
| `seed` | `None` | `int` or `numpy.random.Generator`. A seeded device gives reproducible samples. |
| `c_dtype` | `"complex128"` | Precision: `"complex128"` or `"complex64"`. |
| `backend` | `"numpy"` | The ketjet backend the circuit runs on: `"numpy"`, `"jax"` or a backend instance. See [Backends](#backends). |

## Backends

`backend="jax"` runs the simulation on ketjet's JAX backend (`pip install "ketjet[jax]"`):

```python
import jax

jax.config.update("jax_enable_x64", True)   # needed for the default c_dtype, complex128
dev = qml.device("ketjet.qubit", wires=2, backend="jax")
```

- Results, gradients and seeded samples match the NumPy backend.
- complex128 needs JAX's 64-bit mode. Without it the device raises an error that names the setting; `c_dtype="complex64"`
  works either way.
- PennyLane 0.45's JAX interface (`interface="jax"`) works with JAX below 0.11 only. The ketjet backend itself also
  runs on later versions.
- `diff_method="backprop"` works on the JAX backend: JAX differentiates the simulation itself, under `jax.jit`, to
  any order. See [Backprop](#backprop).
- `diff_method="adjoint"` on the JAX backend gives the same gradients, but its memory grows with circuit depth for
  now. Keep the NumPy backend for deep circuits.

## Operations

ketjet runs these natively:

- **Gates:** `Identity`, `PauliX`, `PauliY`, `PauliZ`, `Hadamard`, `S`, `T`, `SX`, `RX`, `RY`, `RZ`,
  `PhaseShift`, `Rot`, `U3`, `SWAP`, `ISWAP`.
- **Adjoints of fixed gates:** `adjoint(S)`, `adjoint(T)`, `adjoint(SX)`, `adjoint(ISWAP)`.
- **Controlled versions of any of the above**, with any control values. This includes `CNOT`, `CY`, `CZ`,
  `CRX`, `CRY`, `CRZ`, `CRot`, `ControlledPhaseShift`, `Toffoli`, `CSWAP` and `MultiControlledX`.
- **`GlobalPhase`.**

Everything else is **decomposed by PennyLane** into those gates first: templates, `QubitUnitary`, `StatePrep`,
Ising and excitation gates, adjoints of rotations, and so on. An operation with a matrix but no decomposition (for
example `DoubleExcitationPlus` or `BlockEncode`) runs as a `QubitUnitary` of its matrix, which PennyLane can always
decompose.

!!! warning "Trainable operations without a decomposition"
    The decomposition of a large `QubitUnitary` can't be differentiated, so differentiating such an operation's
    parameters on `ketjet.qubit` raises a `DeviceError`. Use them with fixed parameters, or differentiate on
    `default.qubit`.

## Measurements

All of PennyLane's measurements work. ketjet itself computes `expval`, `var` and `probs`, which keeps the
time-critical part of an optimization loop on the backend. `state`, `density_matrix`, `purity`, `vn_entropy`,
`mutual_info` and the other state-based measurements use PennyLane's own definitions on ketjet's state.

With shots, ketjet draws the samples (seeded and reproducible per device seed), and PennyLane turns them into
`expval`, `var`, `probs`, `sample` and `counts`. Shot vectors such as `[100, (50, 2)]` return one result per bin.

## Gradients

| `diff_method` | Supported | Notes |
|---|---|---|
| `"adjoint"` | ✅ | ketjet's adjoint method ([ADR-0004](adr/0004-differentiation.md)): exact, one backend call per circuit, memory independent of depth. Exact `expval` only. |
| `"best"` | ✅ | Resolves to `"adjoint"` for exact expectation values, otherwise to PennyLane's parameter-shift. |
| `"parameter-shift"` | ✅ | PennyLane's transform; works with shots. |
| `"finite-diff"`, `"hadamard"` | ✅ | PennyLane's transforms; `"hadamard"` is exercised by PennyLane's device suite. |
| `"backprop"` | JAX backend | `backend="jax"` with `interface="jax"`, exact results only: see below. |

**Vector-Jacobian products.** Pass `device_vjp=True` to the QNode. Then a gradient of any weighted sum of
expectation values is **one** backward pass, however many expectation values the circuit returns.

### Backprop

On the JAX backend, `diff_method="backprop"` lets JAX differentiate the simulation itself:

```python
import jax

jax.config.update("jax_enable_x64", True)
dev = qml.device("ketjet.qubit", wires=2, backend="jax")

@qml.qnode(dev, interface="jax", diff_method="backprop")
def circuit(x):
    qml.RX(x[0], 0)
    qml.CRY(x[1], [0, 1])
    return qml.expval(qml.PauliZ(1))

jax.jit(jax.grad(circuit))(jax.numpy.array([0.3, 1.2]))
```

- The circuit runs as a compiled program of its parameters, cached on the circuit's structure, so an optimization
  loop compiles once ([ADR-0006](adr/0006-jax-backend.md) D8).
- Any measurement can be differentiated, as can trainable observable coefficients, and `jax.hessian` works.
- Only exact results: samples can't be differentiated, so a circuit with shots raises a `DeviceError`. Use
  parameter-shift with shots.
- Only JAX: backprop with another interface, such as autograd, raises a `DeviceError` that names the fix.
- The NumPy backend has no backprop, since its reference kernels can't be traced, and
  `diff_method="backprop"` there raises `QuantumFunctionError: Device ... does not support backprop`.
- Backprop stores the forward pass, so its memory grows with circuit depth. For deep circuits, the NumPy
  backend's `diff_method="adjoint"` gives the same gradients with memory independent of depth.
- `"best"` still means adjoint for exact expectation values.

### Broadcasting

PennyLane's parameter broadcasting, a gate parameter given as an array of values, works on both backends and
returns what `default.qubit` returns: one result per value.

```python
@qml.qnode(dev, interface="jax", diff_method="backprop")
def circuit(theta):
    qml.RX(theta, 0)
    return qml.expval(qml.PauliZ(0))

circuit(jax.numpy.array([0.1, 0.2, 0.3]))   # shape (3,): one expectation value per angle
```

On the JAX backend, an exact broadcast circuit runs as **one** batched execution, mapped with `jax.vmap` from the
same compilation ([ADR-0007](adr/0007-jax-batching-adjoint-gpu.md) D3), both when it is only executed and under
`diff_method="backprop"`. `qml.Tracker` counts one simulation for it. Everywhere else PennyLane splits it into
one circuit per value (`broadcast_expand`), as its own devices do:

- on the NumPy backend;
- with shots;
- with `diff_method="adjoint"` (and `"best"`) or `"parameter-shift"`. PennyLane's parameter-shift can't
  differentiate with respect to a broadcast parameter on any device; with respect to the others it works.

## Conformance

PennyLane's own device test suite, `pl-device-test`, runs on every pull request in exact mode and in shot mode
(10,000 shots), on the NumPy backend and on the JAX backend. With PennyLane 0.45 and JAX 0.10:

| Backend, mode | Passed | Skipped | Deselected | Expected failures | Failed |
|---|---|---|---|---|---|
| NumPy, exact | 389 | 45 | 0 | 9 | 0 |
| NumPy, shots | 374 | 60 | 0 | 9 | 0 |
| JAX, exact | 394 | 32 | 8 | 9 | 0 |
| JAX, shots | 374 | 60 | 0 | 9 | 0 |

The JAX backend runs the suite's backprop tests under the JAX interface (5 more passing tests). Its 8 deselected
tests are the autograd backprop tests: PennyLane asks a device whether it supports backprop before it knows the
interface, so `ketjet.qubit` must say yes, and these tests then run under autograd, which JAX can't trace. They
fail with the device's `DeviceError` for a non-JAX interface, which is the intended behaviour.

The expected failures are marked as such by PennyLane's suite itself. Every skip is one of these:

| Skipped tests | Why |
|---|---|
| Tests for the other mode (10 exact, 31 shots) | Sampling tests skip in exact mode; analytic-only tests skip with shots. Each runs in the other mode. |
| Tests for PennyLane's old device interface (11 in each mode) | `ketjet.qubit` uses the current `Device` API. |
| Gradient tests for one specific `diff_method` (18 exact on NumPy, 8 exact on JAX, 12 shots) | Mostly backprop, which only the JAX backend supports (see [Backprop](#backprop)); the rest run under the method they target. This includes the suite's JAX gradient tests, which run with JAX installed. |
| `QubitDensityMatrix`, `QutritBasisState` (3 in each mode) | Mixed-state and qutrit operations; `ketjet.qubit` simulates pure qubit states. |
| Torch gradient tests (1 in each mode) | Torch isn't installed in the conformance job. |
| Single-condition checks (2 in each mode) | Conditions that don't apply, e.g. "no warning is raised when shots=None". |

To run the suite yourself:

```bash
uv sync --no-default-groups --group conformance
uv run --no-sync pl-device-test --device=ketjet.qubit
uv run --no-sync pl-device-test --device=ketjet.qubit --analytic=False --shots=10000
JAX_ENABLE_X64=true uv run --no-sync pl-device-test --device=ketjet.qubit --device-kwargs backend=jax \
  -k "not (test_gradients_autograd and backprop)"
uv run --no-sync pytest tests/integration/test_pennylane_jax_interface.py tests/integration/test_pennylane_backprop.py \n  tests/integration/test_pennylane_broadcasting.py
```

The last line runs ketjet's own QNode tests under PennyLane's JAX interface, which the main test suite skips
because it uses the latest JAX.

The conformance environment pins JAX below 0.11, the newest version PennyLane 0.45's JAX interface works with, so it
replaces the development environment. Run `uv sync` afterwards to get that back.

## Limitations

- **Backprop on the JAX backend only**, with a JAX interface and exact results (see [Backprop](#backprop)).
- **Broadcast circuits** run as one batched execution only on the JAX backend, exact, executed or under backprop
  (see [Broadcasting](#broadcasting)). Otherwise they run as one circuit per value.
- **Mid-circuit measurements** are deferred: each one becomes an extra wire.
- **Supported PennyLane versions:** 0.44.1 to 0.45.x. CI runs the minimum and the latest. 0.44.0 isn't supported:
  it can't be imported with current versions of its own dependency `diastatic-malt`, which PennyLane fixed in 0.44.1.
  Newer versions are added once CI passes on them.
- **Operations that run from their matrix** (see [Operations](#operations)) are accurate to about 1e-7 on PennyLane
  0.44, whose `QubitUnitary` decomposition is less precise for three or more qubits, and to 1e-10 on 0.45.
