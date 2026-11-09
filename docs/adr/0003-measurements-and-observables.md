# ADR-0003: Measurements and observables

| | |
|---|---|
| **Status** | Proposed |
| **Date** | 2026-11-10 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-304, QS-305, QS-306; later QS-404 (adjoint), QS-504 (PennyLane), QS-603 (JAX), QS-903 (GPU) |

## Context

Sprint 2 turns the engine into something a user can read results from: probabilities, expectation values,
variances and samples. These outputs are the inputs to every gradient method (E4), the PennyLane device (E5) and
every benchmark (E11), so their representation and semantics must be fixed now. The forces:

- **PennyLane compatibility:** results must equal `default.qubit`'s, including wire order and output shapes.
- **Performance:** expectation values of Pauli strings and Hamiltonians are the hot path of variational
  algorithms (VQE, QML). They must not build $2^n \times 2^n$ matrices.
- **Future backends:** a GPU backend must compute these results **on the device**, transferring only scalars or
  samples (QS-903). The API must not force the state onto the host.
- **Reproducibility:** sampling must be seedable and identical across platforms.

## Decisions

### D1. Observables are immutable values with algebra

`ketjet.observables` provides frozen observable types:

| Type | Meaning | Example |
|---|---|---|
| `Pauli` | Tensor product of Pauli operators on distinct wires (identity elsewhere) | `Z(0) @ Z(1)`, `X(2)` |
| `Hermitian` | Arbitrary Hermitian matrix on given wires (validated Hermitian on construction) | `Hermitian(m, wires=(0, 1))` |
| `Prod` | Tensor product of observables on disjoint wires | `Hermitian(m, (0,)) @ X(1)` |
| `Sum` | Real-weighted sum of observables: a Hamiltonian | `0.5 * Z(0) @ Z(1) - 0.2 * X(0)` |

Operator overloading gives textbook syntax: `@` (tensor product), scalar `*`, `+` and `-`. A product of
Paulis simplifies to one `Pauli`, so `Z(0) @ Z(1)` stays on the fast path (D3). Coefficients are real, so every
observable is Hermitian by construction.

| Option | Pros | Cons |
|---|---|---|
| **A. Typed observable values with algebra** ✅ | Reads like the math; structure (Pauli vs dense) drives fast paths; immutable and hashable like `Operation` | More types than a single matrix class |
| B. Every observable is a dense matrix | One code path | $O(4^n)$ memory for a Hamiltonian on $n$ wires; unusable beyond ~14 qubits |
| C. Strings (`"ZZI"`) | Compact | Wire order ambiguous; no Hermitian or tensor support; error-prone |

### D2. One mechanism: apply the observable, then take inner products

Every observable implements **"apply to a state"**, computing $O|\psi\rangle$. Then for every observable:

$$\langle O \rangle = \operatorname{Re}\,\langle \psi | O \psi \rangle, \qquad \operatorname{Var}(O) = \lVert O\psi \rVert^2 - \langle O \rangle^2 .$$

The variance formula holds for any Hermitian $O$, since $\langle \psi | O^2 | \psi \rangle = \langle O\psi | O\psi \rangle$.
This gives expectation values **and** variances for every observable type, including Hamiltonians, from one
primitive. The same primitive is what the adjoint method needs for its backward pass (QS-404).

### D3. Pauli strings use a matrix-free fast path

A Pauli string acts on a basis state as a phased bit-flip. With `x_mask` the wires carrying X or Y, `z_mask` the
wires carrying Z or Y, and $n_Y$ the number of Ys:

$$P\,|i\rangle = i^{\,n_Y}\,(-1)^{\operatorname{popcount}(i \,\&\, z\_mask)}\;|\,i \oplus x\_mask\,\rangle .$$

So $P|\psi\rangle$ is one gather plus one sign vector: $O(2^n)$ time, no matrices. A Hamiltonian with $T$ Pauli
terms costs $O(T \cdot 2^n)$, which is exactly the cost profile native and GPU kernels will reproduce. `Hermitian`
observables use the `apply_matrix` kernel on their wires.

### D4. Probabilities follow PennyLane's wire order

`probs(wires)` returns the marginal distribution over `wires`, with `wires[0]` as the most significant bit of the
result index (identical to `qml.probs(wires=...)`). Omitting `wires` means all wires in order. The result's real
dtype matches the state's precision (`float64` for complex128, `float32` for complex64), with no silent promotion
(ADR-0001 D3).

### D5. Sampling is seeded, exact and platform-independent

- `sample(shots, wires=None, seed=None)` returns an integer array of shape `(shots, len(wires))` of 0/1 bits,
  matching PennyLane's layout.
- Randomness comes from a NumPy `Generator` (PCG64). `seed` accepts an `int`, a `Generator` or `None`. The same
  seed gives the same samples on every OS and Python version.
- Probabilities are accumulated in `float64` and renormalized before sampling, so complex64 rounding can't bias
  or break the draw.
- **Shot-based expectation values** rotate each observable into its eigenbasis and average the sampled
  eigenvalues. For a Pauli string, X uses H and Y uses S†·H, and the eigenvalue is the parity of the measured bits.
  `Hermitian` uses its eigendecomposition. A `Sum` samples each term with the full shot budget. This mirrors
  `default.qubit` without term grouping, which is a later optimization.

### D6. Measurements are part of the backend protocol

`Backend` (ADR-0001 D4) gains `probs`, `expval`, `var` and `sample`. These are **read-only** (they don't consume the
state). The NumPy backend delegates to the reference functions in `ketjet.measurements`. A GPU backend
implements them on the device and returns only scalars or samples, which is what QS-903 requires.

| Option | Pros | Cons |
|---|---|---|
| **A. Measurements in the protocol** ✅ | GPU and JAX backends keep states on the device; one test suite per backend | Protocol grows by four methods |
| B. Measure via `to_numpy()` | No protocol change | Every measurement copies $2^n$ amplitudes to the host, defeating the GPU backend |

### D7. User-facing API

The `Circuit` builder (ADR-0002 D6) runs on a backend (default: `NumPyBackend`) through convenience methods:
`circuit.state()`, `circuit.probs(wires)`, `circuit.expval(obs, shots=None, seed=None)`, `circuit.var(obs)` and
`circuit.sample(shots, wires, seed)`. Circuits with unbound symbolic parameters raise a clear error.

## Consequences

- **Positive:** VQE-style Hamiltonians are cheap from day one. One primitive serves expectation values, variances
  and (in Sprint 3) adjoint gradients. GPU backends will avoid host transfers. Results match PennyLane exactly.
- **Negative:** four more protocol methods for every backend. The Pauli fast path needs its own tests against the
  dense definition.
- **Follow-ups:** QS-305 (observables), QS-304 (exact measurements), QS-306 (sampling), QS-404 (adjoint reuses D2),
  QS-903 (device-side measurements).

## Validation

- Pauli fast path equals the dense Kronecker-product matrix on random states and random words (property test).
- `probs`, `expval` and `var` equal `default.qubit` on random circuits and observables, both precisions.
- Sampling: same seed gives identical samples; a chi-square test against `probs` passes at a fixed significance;
  shot-based expectation values converge within $3\sigma$ of the exact value.
