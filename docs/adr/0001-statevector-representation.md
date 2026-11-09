# ADR-0001: Statevector representation, qubit ordering and precision policy

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-11-09 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-201, QS-202, QS-203, QS-209 |

## Context

Every backend (NumPy, JAX, C++, CUDA) stores and transforms an $n$-qubit statevector of $2^n$ complex amplitudes.
Before any kernel is written, we must fix:

1. how amplitudes are laid out in memory,
2. which qubit corresponds to which bit of the basis index,
3. which floating-point precisions are supported and how they interact,
4. who owns a state buffer when a backend transforms it (pure functional vs in-place),
5. how a future batch dimension fits in without breaking the above.

These choices are hard to change later: they leak into kernels, tests, the PennyLane adapter and the benchmarks.
The forces are **PennyLane compatibility** (our main integration target), **performance** on native and GPU backends
(where memory is the scarce resource, since one 30-qubit complex64 state is 8 GB), and **JAX's immutable arrays**.

## Decisions

### D1. Memory layout: a flat, contiguous array, with a tensor view for kernels

The canonical storage is a **contiguous, C-ordered 1-D array of length $2^n$**. NumPy and JAX kernels operate on a
zero-copy **reshape to a rank-$n$ tensor of shape `(2,) * n`**, so applying a $k$-qubit gate is a tensor contraction
over $k$ axes.

| Option | Pros | Cons |
|---|---|---|
| **A. Flat 1-D storage + `(2,)*n` view** ✅ | Matches PennyLane's `state()` output and the flat buffers C++/CUDA kernels need; the reshape is free | Kernels must reshape at the boundary |
| B. Always a rank-$n$ tensor | Natural for `tensordot` | Shape depends on $n$, which is awkward for native code, typing and JAX tracing |
| C. Flat storage, bit-manipulation loops everywhere | Same model as the native kernels | Slow and unreadable in NumPy, which defeats the point of the reference backend |

### D2. Qubit ordering: wire 0 is the most significant bit (big-endian)

For the basis state $|b_0 b_1 \dots b_{n-1}\rangle$, the index is

$$\text{index} = \sum_{w=0}^{n-1} b_w \, 2^{\,n-1-w}.$$

So **wire $w$ is axis $w$** of the tensor view. A $k$-qubit gate matrix acting on wires $(w_0, \dots, w_{k-1})$ uses
the same convention, with $w_0$ as its most significant bit, so `CNOT` on wires `(0, 1)` is the textbook matrix.

| Option | Pros | Cons |
|---|---|---|
| **A. Big-endian (wire 0 = MSB)** ✅ | Identical to PennyLane: integration and differential tests need no reindexing; textbook matrices read naturally | Differs from Qiskit (little-endian), which matters only for benchmarks |
| B. Little-endian (wire 0 = LSB) | Matches Qiskit and many bit-twiddling kernels | Every PennyLane interaction needs an axis reversal, a permanent source of bugs |

Qiskit interop (benchmarks only, Sprint 7) reverses the axes in one tested helper.

### D3. Precision: complex128 by default, complex64 opt-in, never mixed

- Supported dtypes: **`complex128`** (default) and **`complex64`**. A state's dtype is fixed when it is allocated.
- Gate matrices are **cast to the state's dtype** before application. A complex64 state is never silently promoted to
  complex128, which would double memory and hide precision bugs.
- Test tolerances are defined once per precision in `tests/tolerances.py` (plan.md §6.3).

| Option | Pros | Cons |
|---|---|---|
| **A. complex128 default, complex64 opt-in** ✅ | Correct-first default that matches `default.qubit`; complex64 halves memory and doubles bandwidth for GPU/large runs | Two precisions to test |
| B. complex64 default | Faster by default | Surprising precision loss for users; gradient checks get noisier |
| C. complex128 only | Simplest | Gives up the precision/performance trade-off that matters most on GPUs |

### D4. Ownership: value-semantic API, consumed inputs

Backend operations have **functional signatures** (`new_state = backend.apply_operation(state, op)`), and the caller
must **treat the input state as consumed**: it must not be read or reused after the call.

- The NumPy reference backend is **pure**: it returns a new array and never mutates its input.
- JAX is naturally pure, since its arrays are immutable.
- Native and CUDA backends may **update in place** and return the same buffer, avoiding a second $2^n$ allocation.

| Option | Pros | Cons |
|---|---|---|
| **A. Value-semantic signature, consumed input** ✅ | One protocol fits immutable (JAX) and in-place (C++/CUDA) backends; native code avoids 2× peak memory | Callers must follow the "don't reuse the input" rule. Mitigation: the cross-backend tests use only return values |
| B. Always pure (copy on every gate) | Simplest mental model | Doubles peak memory on native/GPU, which costs a qubit at the memory limit |
| C. Always in-place (`apply_(state, op) -> None`) | No copies | Incompatible with JAX; forces two protocols |

### D5. Batching: reserved leading axis (implemented in Sprint 6)

A batched state will have shape **`(B, 2**n)`**, with the batch on the **leading axis**, so its tensor view is
`(B,) + (2,) * n` and wire $w$ is axis $w + 1$. Sprint 1 kernels don't implement batching, but they're written in
terms of "the last $n$ axes", so adding a leading batch axis is an extension, not a rewrite. This lines up with
`jax.vmap` (QS-605) and PennyLane parameter broadcasting (QS-506).

### D6. Sizes and validation

- `n_wires` must be an integer with $1 \le n \le 63$. Larger values can't index a 64-bit address space.
- A helper reports memory up front: `state_nbytes(n, dtype) = 2**n * itemsize` (16 bytes for complex128, 8 for complex64).
- Allocation accepts an optional `max_bytes`. If the state would exceed it, or the allocation itself fails, it raises
  a `MemoryError` stating the qubit count, dtype and required bytes, instead of NumPy's generic error.

## Consequences

- **Positive:** PennyLane integration and differential tests compare arrays directly. One backend protocol covers
  every backend. Memory-critical backends can work in place. Batching has a defined home.
- **Negative:** callers must respect the consumed-input rule. Two precisions double the precision-specific tests.
- **Follow-ups:** QS-202 (allocation and memory helper), QS-203 (`apply_matrix` on the tensor view), QS-209 (backend
  protocol encoding D4), QS-605/QS-506 (batching per D5).

## Validation

- Property tests: every gate preserves the norm, and applying a gate then its inverse returns the original state,
  in both precisions.
- Differential tests match PennyLane gate matrices and `default.qubit` states without any reindexing (proves D2).
- A test asserts that a complex64 state stays complex64 through every gate (proves D3).
