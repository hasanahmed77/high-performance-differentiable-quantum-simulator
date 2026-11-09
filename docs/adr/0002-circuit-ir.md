# ADR-0002: Circuit IR (operations, gate specifications, parameters, wires)

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-11-09 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-301, QS-204, QS-209, later QS-302/303 (API), QS-402/404 (gradients), QS-802–805 (fusion) |

## Context

The circuit intermediate representation (IR) is the contract between the user API, the compiler passes (decomposition,
**gate fusion**), the **differentiation** methods (parameter-shift, adjoint) and every backend. Three later epics depend
on it:

- **Fusion (E8)** must inspect and rewrite circuits and keep cheap gate structure (diagonal, permutation, controlled).
- **Differentiation (E4)** must know which parameters are trainable and each gate's generator or shift rule.
- **Backends (E2, E6, E7, E9)** must dispatch each operation to the fastest kernel.

So the IR must be **inspectable data**, carry **gate structure explicitly**, and be **safe to share** between passes.

## Decisions

### D1. An operation is an immutable value

```python
@dataclass(frozen=True, slots=True)
class Operation:
    gate: str                          # key into the gate registry, e.g. "RX"
    wires: tuple[int, ...]             # target wires
    params: tuple[ParamLike, ...] = () # floats or symbolic Parameters
    controls: tuple[int, ...] = ()     # control wires (disjoint from targets)
    control_values: tuple[bool, ...] = ()  # per-control trigger value; default all True
```

| Option | Pros | Cons |
|---|---|---|
| **A. Frozen dataclass** ✅ | Hashable and comparable; safe to share between passes without defensive copies; cheap snapshots | Rewriting means building new ops (cheap, since they're tiny) |
| B. Mutable gate objects (classic OOP class per gate) | Familiar | Passes can corrupt each other's circuits; aliasing bugs; harder to cache |
| C. Raw tuples | Fastest to build | No validation or names; unreadable |

### D2. Controls are structure, not separate gate names

Controlled gates are represented as **base gate + control wires**: `CNOT(c, t)` is `Operation("X", wires=(t,), controls=(c,))`,
Toffoli is `X` with two controls, and `CRX(θ)` is `RX(θ)` with one control. Named constructors (`CNOT`, `CZ`, `Toffoli`,
`CRX`, …) build these canonical forms, and `matrix()` can still produce the full matrix (for example to compare
against PennyLane).

| Option | Pros | Cons |
|---|---|---|
| **A. Canonical base + controls** ✅ | Kernels touch only the control-satisfied subspace (half the work per control); fusion and differentiation reason about one base gate; any gate becomes controllable (QS-206) for free | The name a user sees (`CNOT`) differs from the stored base (`X`); the reverse mapping is kept for display |
| B. Every controlled gate is its own named gate | Mirrors textbooks | Combinatorial gate list; kernels and gradient rules duplicated per variant |

### D3. A registry of gate specifications carries the structure

Each gate name maps to an immutable `GateSpec`:

| Field | Purpose | Used by |
|---|---|---|
| `name`, `num_wires`, `num_params` | Validation | IR construction |
| `matrix(params, dtype)` | Dense unitary of the base gate | Reference kernel, tests, fusion |
| `is_diagonal`, `is_permutation` | Selects fast kernels; fusion keeps these cheap (QS-804) | QS-205, QS-706, QS-804 |
| `is_self_inverse`, `inverse(params)` | Adjoint pass, inverse tests | QS-404 |
| `generator` / shift rule *(added in Sprint 3)* | Parameter-shift and adjoint gradients | QS-402, QS-404 |

Structure flags are **verified by tests against the matrix** (a gate claiming `is_diagonal` must have a diagonal matrix
for random parameters), so a wrong flag can't silently produce wrong results through a fast path.

### D4. Parameters: floats or named symbols

`ParamLike = float | Parameter`, where `Parameter` is a frozen, named symbol (`Parameter("theta")`). Binding values
produces a concrete circuit (QS-303). Trainable-parameter tracking (which parameters a gradient is taken with respect
to) is derived from symbols and arrives with QS-303.

### D5. Wires are integers `0 … n-1`

The core uses **dense integer wires only**. Arbitrary wire labels (strings, non-contiguous ints) are mapped to
integers once, in the PennyLane adapter (QS-502). That keeps every kernel and pass free of label lookups.

### D6. Circuits: a fluent builder over immutable operations

The user-facing `Circuit(n_wires)` is a **mutable, append-only builder** with fluent methods (`c.h(0).cnot(0, 1)`)
that validate each operation as it's added. `circuit.operations` returns a **tuple snapshot**. Since operations are
immutable, snapshots are cheap and safe. **Compiler passes are pure functions**:
`Sequence[Operation] -> tuple[Operation, ...]`.

| Option | Pros | Cons |
|---|---|---|
| **A. Mutable builder, immutable ops, pure passes** ✅ | Ergonomic API; passes can't corrupt the user's circuit; each pass is testable in isolation | Two concepts (builder vs op tuple) |
| B. Fully immutable `Circuit` (every append returns a new circuit) | Purest | O(n²) copying while building large circuits, or a persistent data structure, which is over-engineering |
| C. Mutable everything | Simplest | Passes mutating shared state is exactly the bug class fusion would introduce |

### D7. Validate at construction (fail fast)

Adding an operation checks that: the gate exists; the number of wires and parameters matches the spec; all wires are
in `0 … n-1`; targets and controls are disjoint and free of duplicates; and `control_values` has one entry per
control. Invalid circuits never reach a backend, so kernels can assume valid input.

## Consequences

- **Positive:** fusion, differentiation and kernel dispatch read structure from data instead of re-deriving it.
  Passes are pure and composable. Controlled gates are cheap everywhere.
- **Negative:** a registry indirection on every operation (negligible next to $2^n$ work). The display name and the
  stored base differ for controlled gates.
- **Follow-ups:** QS-204 implements `Operation`, `GateSpec` and the standard registry. QS-302/303 build the `Circuit`
  API and parameters on top. QS-402/404 extend `GateSpec` with generators. QS-802 builds the pass framework.

## Validation

- Every gate's matrix is unitary and matches PennyLane's matrix for random parameters (differential test).
- Every structure flag is verified against its matrix.
- Invalid operations raise `ValueError` with a specific message (one test per validation rule).
