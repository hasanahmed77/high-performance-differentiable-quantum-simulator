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
| `"backprop"` | ❌ | Not yet: see below. |

**Vector-Jacobian products.** Pass `device_vjp=True` to the QNode. Then a gradient of any weighted sum of
expectation values is **one** backward pass, however many expectation values the circuit returns.

**Backprop.** `diff_method="backprop"` raises `QuantumFunctionError: Device ... does not support backprop`.
Backprop needs the simulation itself to be traceable by an autodiff framework, and ketjet's NumPy reference kernels
deliberately aren't. **Use `diff_method="adjoint"` instead:** it gives the same gradients, exactly, with much less
memory for deep circuits. Backprop on the device is planned for the JAX backend (QS-608).

## Conformance

PennyLane's own device test suite, `pl-device-test`, runs on every pull request in exact mode and in shot mode
(10,000 shots), on the NumPy backend and on the JAX backend. With PennyLane 0.45 and JAX 0.10, the results are the
same on both backends:

| Mode | Passed | Skipped | Expected failures | Failed |
|---|---|---|---|---|
| Exact | 389 | 45 | 9 | 0 |
| Shots | 374 | 60 | 9 | 0 |

The expected failures are marked as such by PennyLane's suite itself. Every skip is one of these:

| Skipped tests | Why |
|---|---|
| Tests for the other mode (10 exact, 31 shots) | Sampling tests skip in exact mode; analytic-only tests skip with shots. Each runs in the other mode. |
| Tests for PennyLane's old device interface (11 in each mode) | `ketjet.qubit` uses the current `Device` API. |
| Gradient tests for one specific `diff_method` (18 exact, 12 shots) | Mostly backprop, which isn't supported yet (see [Gradients](#gradients)); the rest run under the method they target. This includes the suite's JAX gradient tests, which run with JAX installed. |
| `QubitDensityMatrix`, `QutritBasisState` (3 in each mode) | Mixed-state and qutrit operations; `ketjet.qubit` simulates pure qubit states. |
| Torch gradient tests (1 in each mode) | Torch isn't installed in the conformance job. |
| Single-condition checks (2 in each mode) | Conditions that don't apply, e.g. "no warning is raised when shots=None". |

To run the suite yourself:

```bash
uv sync --no-default-groups --group conformance
uv run --no-sync pl-device-test --device=ketjet.qubit
uv run --no-sync pl-device-test --device=ketjet.qubit --analytic=False --shots=10000
JAX_ENABLE_X64=true uv run --no-sync pl-device-test --device=ketjet.qubit --device-kwargs backend=jax
```

The conformance environment pins JAX below 0.11, the newest version PennyLane 0.45's JAX interface works with, so it
replaces the development environment. Run `uv sync` afterwards to get that back.

## Limitations

- **No backprop** yet, on either backend (see [Gradients](#gradients)).
- **Batched parameters** run as a loop of circuits (PennyLane's `broadcast_expand`), until native broadcasting.
- **Mid-circuit measurements** are deferred: each one becomes an extra wire.
- **Supported PennyLane versions:** 0.44 and 0.45. Newer versions are added once CI passes on them.
