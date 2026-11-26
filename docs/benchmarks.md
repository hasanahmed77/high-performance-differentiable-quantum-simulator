# Benchmarks

ketjet's benchmark suite times five kinds of work on the NumPy and JAX backends and writes the results, with a
record of the machine, to `benchmarks/results/`. How it measures is decided in
[ADR-0008](adr/0008-benchmark-methodology.md). Comparisons with other simulators (Sprint 7, QS-1103) and plots
(QS-1104) build on it.

## The benchmarks

| Name | What is timed | Sizes |
|---|---|---|
| `gate-h` | 10 layers of Hadamards on every qubit, a dense single-qubit gate | 4 to 20 qubits |
| `gate-rz` | 10 layers of `RZ` on every qubit, a diagonal gate | 4 to 20 qubits |
| `gate-cnot` | 10 layers of CNOTs on neighbouring pairs, a permutation | 4 to 20 qubits |
| `qft` | The quantum Fourier transform: Hadamards, controlled phases and SWAPs | 4 to 20 qubits |
| `random` | 8 layers of a random rotation per qubit and CNOTs on random pairs | 4 to 20 qubits |
| `vqe-grad` | The gradient of a transverse-field Ising energy, for a 4-layer hardware-efficient ansatz | 4 to 20 qubits |
| `qml-batch` | The gradient of a classifier's mean output over a batch of 32 inputs, with respect to its weights | 4 to 16 qubits |

The state benchmarks start with a layer of `RY` rotations whose angles are inputs, so that no compiler can work out
the answer ahead of time. Every circuit and angle is seeded, so every run and every library gets the same ones.

Each benchmark runs on these **runners**:

- `ketjet-numpy`: the NumPy backend, as `Circuit.state()` or `gradients.grad(..., method="adjoint")`. Batches are
  a Python loop.
- `ketjet-jax`: the JAX backend, compiled, with JAX's own reverse mode for gradients and `jax.vmap` for batches.
- `ketjet-jax-adjoint`: the same with `diff_method="adjoint"`, for the two gradient benchmarks.

## How it measures

Following ADR-0008:

- **The first call is reported separately.** For JAX it includes the compilation, which can take seconds. The
  headline time is the steady state, the time of later calls.
- **Every answer is checked first.** The first call's result is compared with the NumPy backend's in complex128,
  at the test suite's tolerances. A wrong answer stops the run; it is never timed.
- **Two warm-up calls, then samples of at least 0.2 s.** Fast calls are repeated within a sample. There are 15
  samples, or 5 for a call over 10 s.
- **Each case runs in three fresh processes.** The reported time is the median of the three medians, with the
  interquartile range. The spread between the processes is stored too.
- **JAX results are synchronized** with `block_until_ready`, so a GPU's time is the time until it has finished.
- **GPU calls under 10 ms are marked dispatch-bound,** because their timings are mostly kernel launches.
- **A ratio between 0.9 and 1.1 is "no clear difference",** not a win or a loss.

## Running it

In the GPU workstation image ([GPU workstation](dev/gpu-workstation.md)), from the repository:

```bash
uv run python -m benchmarks.run                    # everything on the CPU: about 40 minutes
uv run python -m benchmarks.run --device gpu       # the JAX runners on the GPU
uv run python -m benchmarks.run --benchmarks qft,vqe-grad --sizes 8,12 --quick
```

`--quick` (one process, five samples) and `--smoke` (one call, 3 qubits) are for development. They are written as
`unpublished-*.json`, which git ignores. CI runs every benchmark at 3 qubits on every pull request
(`tests/integration/test_benchmark_suite.py`), so the suite can't stop working unnoticed.

While a run is going, the machine should be plugged in and otherwise idle: no games, video or other heavy programs
(ADR-0008 D6).

## GPU capacity (2026-11-26)

How many qubits the JAX backend fits on the RTX 4060 (8 GB), measured with
`uv run python -m benchmarks.capacity --device gpu` (QS-1108). Each size runs in its own process, growing until
one runs out of memory; the full record, with the peak memory of every size, is in
`benchmarks/capacity/2026-11-26-gpu.json`.

| Workload | complex64 | complex128 | Peak memory, in copies of the state |
|---|---|---|---|
| The `random` circuit's final state | 25 qubits | 24 qubits | 8 to 9 |
| The `vqe-grad` gradient, adjoint | 25 qubits | 24 qubits | 4 to 5 |
| The `vqe-grad` gradient, backprop | 21 qubits | 20 qubits | about 170, and growing with depth |

What it shows:

- **The forward pass uses 8 to 9 states of memory,** where in-place kernels would need one or two. At 26 qubits
  in complex64 a 512 MB state asked for a 4 GB buffer. XLA keeps intermediate copies between gates; this is the
  same overhead that keeps the GPU far from its memory bandwidth (above), and it's the CUDA backend's job to
  remove (Sprints 12–13). plan.md §4.5 planned 29 and 28 qubits; it now lists these measured numbers too.
- **Adjoint fits as many qubits as the forward pass,** using about half its memory at the largest sizes.
- **Backprop is limited by depth:** it keeps every intermediate state, so at 4 layers it fits 4 qubits fewer than
  adjoint, and a deeper circuit fits fewer still. Adjoint is the method for large circuits (ADR-0007).
- **complex128 is slow on this GPU:** the RTX 4060, like other consumer cards, runs double precision at a small
  fraction of its single-precision rate. The adjoint gradient at 24 qubits took 37 s in complex128 against 2.8 s
  in complex64.

## The result files

Each run writes one JSON file, `benchmarks/results/<date>-<device>-<precision>.json`. It holds:

- `machine`: the date; the CPU, cores, memory, GPUs and NVIDIA driver; the OS, kernel and WSL version, Python, the
  git commit, the Docker image ID and every library's version; thread and device environment variables, and the
  memory cap.
- `settings`: the sample settings used, and whether they're the published ones.
- `cases`: one entry per benchmark, runner and size, with the median, quartiles, minimum, first call, gate and
  parameter counts, and every process's own samples.

A file without its machine record is rejected (`benchmarks.harness.validate`).

## First results (2026-11-24)

On the GPU workstation: an Intel Core i5-9600K (6 cores), 6 GB for the container, an RTX 4060 (8 GB, driver
610.88), JAX 0.12.0.1, complex128, at commit `199d1ee`. The steady-state median of three processes, in
milliseconds, at the largest size of each benchmark. The full numbers, including the first calls and every size,
are in `benchmarks/results/2026-11-24-cpu-complex128.json` and `2026-11-24-gpu-complex128.json`.

| Benchmark | Qubits | NumPy | JAX, CPU | JAX adjoint, CPU | JAX, GPU | JAX adjoint, GPU |
|---|---|---|---|---|---|---|
| `gate-h` | 20 | 2,364 | 1,042 | | 922 | |
| `gate-rz` | 20 | 1,001 | 160 | | 97 | |
| `gate-cnot` | 20 | 539 | 429 | | 281 | |
| `qft` | 20 | 1,047 | 826 | | 218 | |
| `random` | 20 | 1,842 | 862 | | 661 | |
| `vqe-grad` | 20 | 7,091 | 2,649 | 3,598 | 1,068 | 1,891 |
| `qml-batch` | 16 | 4,849 | 2,258 | 3,188 | 511 | 591 |

What they show:

- **Compiled JAX beats NumPy once it has compiled,** by 1.3 to 6.3 times at 20 qubits on the CPU and by much more
  on small circuits, where NumPy's per-gate Python overhead dominates. The first call, JAX's compilation, takes
  1 to 24 seconds, so a circuit that runs only a few times is faster on NumPy.
- **The GPU helps most where there's the most work per call:** 4.8 times NumPy for the QFT, 6.6 times for the VQE
  gradient and 9.5 times for the batched QML gradient. 26 of the 43 GPU cases take under 10 ms and are marked
  dispatch-bound.
- **Backprop is faster than adjoint** at these depths, by 1.4 times on the CPU and up to 1.8 times on the GPU.
  Adjoint's advantage is memory (ADR-0007), not time.
- **The GPU is far from its memory bandwidth.** A layer of 20 Hadamards on 20 qubits takes about 4 ms per gate on
  the RTX 4060, about 30 times longer than reading and writing the 16 MB state once. The CUDA backend
  (Sprints 12–13) and the roofline analysis (QS-1106) are where that gets measured and fixed.
- **Noise:** the median spread between the three processes was 1.8% on the CPU and 1.4% on the GPU. Nine CPU
  cases spread by more than 5%, all of them JAX cases, with the largest 21% (`vqe-grad`, adjoint, 16 qubits).

Comparisons with other simulators come next (QS-1103).
