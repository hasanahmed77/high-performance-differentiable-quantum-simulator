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
| `backend` | `NumPyBackend()` | The ketjet backend the circuit runs on. |

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
memory for deep circuits. Backprop through ketjet comes with its JAX backend.

## Conformance

PennyLane's own device test suite, `pl-device-test`, runs on every pull request in exact mode and in shot mode
(10,000 shots). With PennyLane 0.45:

| Mode | Passed | Skipped | Expected failures | Failed |
|---|---|---|---|---|
| Exact | 380 | 37 | 9 | 0 |
| Shots | 365 | 52 | 9 | 0 |

The expected failures are marked as such by PennyLane's suite itself. Every skip is one of these:

| Skipped tests | Why |
|---|---|
| Tests for the other mode (10 exact, 31 shots) | Sampling tests skip in exact mode; analytic-only tests skip with shots. Each runs in the other mode. |
| Tests for PennyLane's old device interface (11 in each mode) | `ketjet.qubit` uses the current `Device` API. |
| Gradient tests for one specific `diff_method` (9 exact, 3 shots) | Mostly backprop, which isn't supported yet (see [Gradients](#gradients)); the rest run under the method they target. |
| `QubitDensityMatrix`, `QutritBasisState` (3 in each mode) | Mixed-state and qutrit operations; `ketjet.qubit` simulates pure qubit states. |
| JAX and Torch gradient tests (2 in each mode) | Not installed in the conformance job. JAX is tested with the JAX backend: PennyLane 0.45 doesn't yet support JAX 0.12. |
| Single-condition checks (2 in each mode) | Conditions that don't apply, e.g. "no warning is raised when shots=None". |

To run the suite yourself:

```bash
uv sync --group conformance
uv run pl-device-test --device=ketjet.qubit
uv run pl-device-test --device=ketjet.qubit --analytic=False --shots=10000
```

## Limitations

- **No backprop**, until the JAX backend (see [Gradients](#gradients)).
- **Batched parameters** run as a loop of circuits (PennyLane's `broadcast_expand`), until native broadcasting.
- **Mid-circuit measurements** are deferred: each one becomes an extra wire.
- **Supported PennyLane versions:** 0.44 and 0.45. Newer versions are added once CI passes on them.
