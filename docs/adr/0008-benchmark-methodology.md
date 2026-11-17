# ADR-0008: Benchmarks time the steady state, report medians, and record the machine

| | |
|---|---|
| **Status** | Accepted: approved by the PO on 2026-11-18 in #155, including D7's 10% starting threshold |
| **Date** | 2026-11-18 |
| **Deciders** | Product Owner / Architect |
| **Related stories** | QS-1101 (this ADR); QS-1102 (suite), QS-1103 (comparisons), QS-1104 (plots), QS-1105 (asv), QS-1106 (roofline), QS-908 (T4 run) |

## Context

Sprint 7 publishes ketjet's first performance numbers against `default.qubit`, `lightning.qubit`,
`lightning.gpu`, Qiskit Aer and Qulacs (plan.md G3, G4), including where ketjet loses. Numbers like that are
only worth publishing if someone else can rerun them and get the same answer, and if a reader can tell a real
difference from noise. Before the suite is written, we need to decide what a benchmark times, how often, which
statistic is reported, what a comparison must hold equal, and what is recorded about the machine.

The forces:

- **JAX compiles on the first call.** A benchmark that includes it measures the compiler, not the simulator.
  One that hides it flatters JAX for code that runs once.
- **GPUs run asynchronously.** Without a sync, a timer measures how fast work is queued.
- **There is one benchmark machine.** The RTX 4060 workstation is the only machine in use: CPU and GPU numbers
  both come from it, in the pinned `ketjet-gpu` image under WSL2 (plan.md §4.5). GPU runs are by hand, never in
  CI (ADR-0007 D5, amended).
- **plan.md §6.1 sets a 5% regression threshold** for the nightly asv run on hosted runners.

### What the planning spike measured

On the workstation (Intel Core i5-9600K, 6 cores; RTX 4060; 8 GB for WSL), in the `ketjet-gpu` image (JAX
0.12.0.1), timing `RY` + `CNOT` layers (4 layers, complex128). Each case ran in **5 separate processes**, each
taking 15 samples of at least 0.2 s:

| Case | First call | Later calls (median) | Spread within one process (IQR / median) | Spread of the 5 processes' medians |
|---|---|---|---|---|
| NumPy, 12 qubits | 4.7 ms | 4.4 to 4.5 ms | 1.3 to 2.2% | **3.7%** |
| NumPy, 20 qubits | 1.09 s | 1.10 to 1.12 s | 1.3 to 10.2% | **2.2%** |
| JAX CPU, 12 qubits | 0.54 to 0.84 s | 1.96 to 1.98 ms | 0.3 to 0.8% | **1.3%** |
| JAX CPU, 20 qubits | 1.8 s | 0.71 to 0.74 s | 0.7 to 8.0% | **3.4%** |
| JAX GPU, 12 qubits | 1.4 to 1.9 s | 2.6 to 3.1 ms | 0.7 to 6.6% | **19%** |
| JAX GPU, 20 qubits | 3.9 to 4.0 s | 0.54 to 0.56 s | 0.2 to 1.3% | **4.0%** |

What this shows:

- **The first call is a different quantity.** JAX's first call is about 300 to 700 times slower than later calls at
  12 qubits, and 2.5 to 7.5 times at 20. NumPy's first call is within 10% of later calls.
- **Single samples are noisy; medians are not.** One sample ran up to 20% slower than its process's median,
  but the median moved by at most 4% between processes for every CPU case and for the GPU at 20 qubits.
- **Small GPU circuits are dispatch-bound and noisy.** At 12 qubits the GPU is slower than the CPU and its
  medians spread by 19% between processes: the time is mostly launching kernels, and the GPU's clocks vary.
- **5% is at the noise level.** Two runs of unchanged code on a quiet machine already differ by up to 4%.
  Hosted runners, which share hardware, will be noisier; QS-1105 measures by how much.

## Options considered

### Option A: asv for everything
- **Pros:** one tool; regression tracking and a dashboard built in; handles warm-up and repeats.
- **Cons:** built to compare commits of one project, not six libraries; no separate first-call time; no GPU
  sync; its results format has no place for the GPU, driver and image record of D6.

### Option B: pytest-benchmark
- **Pros:** runs with the test suite; JSON output; calibrates rounds.
- **Cons:** the same gaps as asv for first calls, GPU sync and the hardware record; runs every library in one
  process, so one library's threads and memory affect the next.

### Option C: a small harness for the published numbers, asv for regressions (chosen)
- **Pros:** the harness does exactly D1 to D6: first call apart, sync, one process per library, the machine
  record in the JSON. asv keeps doing what it's good at, comparing commits of ketjet.
- **Cons:** about 200 lines of harness to write and test (QS-1102); two tools.

## Decision

**Option C.** The published numbers come from `benchmarks/`' own harness, which follows D1 to D6; asv tracks
ketjet's regressions nightly (D7). The main reason is that a cross-library comparison must hold the machine,
the precision and the process equal and record them, and neither tool does that.

### D1. What is timed

- **The steady state, separately from the first call.** Every case records its first call's time (for JAX,
  the compilation) and, separately, the time of later calls. Plots and tables show the steady state; the first
  call is a separate column, never left out.
- **One call as a user makes it.** A ketjet state is `run(x)` of a compiled circuit, or `circuit.state()` on
  NumPy; a QNode is called as a QNode. Building the circuit is outside the timer unless the benchmark is about
  building circuits.
- **Synchronized.** JAX results are timed with `.block_until_ready()`, so the timer stops when the GPU
  finishes. Results stay on their device: copying them to the host is timed only where a user's call does it
  (a PennyLane device returns NumPy arrays, so its time includes the copy).
- **Correct first.** Before timing, every case checks its result against the NumPy oracle at plan.md §6.3's
  tolerances. A library that gives a wrong answer isn't timed.

### D2. Warm-up

The first call is recorded (D1), then two more calls are made and discarded before sampling. The spike's
second and later calls were within the run's own noise for every case.

### D3. Samples and repeats

- **A sample lasts at least 0.2 s.** Fast calls are repeated within a sample, as many times as fit in 0.2 s
  (measured from the warm-up), and the sample's time is divided by that number.
- **At least 15 samples per case,** or at least 5 for a call that takes over 10 s.
- **One process per library and size.** Each case runs in a fresh subprocess, so threads, caches and memory
  from one library don't affect the next.
- **Headline numbers repeat the whole run in 3 processes**, and the result is the median of the 3 medians. The
  spike's medians moved by up to 4% between processes; one process alone can't show that.

### D4. Statistics and how a difference is reported

- **The median** of the samples is the reported time, with the **interquartile range** as its spread. The
  minimum is stored in the JSON too, because it is what `timeit` users expect, but it isn't the headline: it
  hides how often a run is slow.
- **A speedup is the ratio of two medians.** A ratio between 0.9 and 1.1 is reported as **"no clear
  difference"**, not as a win or a loss: the spike's noise between processes reached 4% per side.
- **Small GPU cases are labelled dispatch-bound** when one call takes less than 10 ms, since their spread
  reached 19% in the spike.
- Plots show the median with IQR bars, on log axes for time against qubits (QS-1104).

### D5. A fair comparison

- **The same circuit**, built from one spec in each library's own API, with the same gates; a library that
  decomposes a gate differently does so in its own time.
- **The same precision.** complex128 by default; complex64 in its own table, only for libraries that support
  it.
- **The same threads.** Every library gets all cores, and the number is recorded; single-thread results are a
  separate table where kernels are compared (QS-1106).
- **Each library as its documentation recommends**, with its defaults otherwise; options used are recorded.
  Versions are pinned in a lock group for the comparison libraries, so a rerun installs the same ones.
- **Losses are published** in the same table as wins (plan.md G3).

### D6. The machine record

Every result file is JSON, committed under `benchmarks/results/`, and starts with a record of where it ran:

- **Hardware:** CPU model, cores and threads, RAM, GPU model and memory, NVIDIA driver;
- **Software:** OS and kernel (and WSL version), Python, ketjet version and git commit (and whether the tree
  was clean), every library's version, CUDA runtime, the Docker image ID;
- **Settings:** thread-count variables (`OMP_NUM_THREADS`, `XLA_FLAGS` and the like), JAX's x64 mode and
  default device, the memory cap, the date.

A result without its record isn't published. Results from a different machine (a Kaggle T4, QS-908) carry
their own record and go in their own table; numbers from two machines are never compared in one plot.

**Running conditions** go into the record and the docs: the workstation plugged in on Windows' "Best
performance" power mode, no other heavy programs running (games, browsers playing video), and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so the measured memory is ketjet's.

### D7. Regression tracking (QS-1105)

- **asv compares commits within one job** (`asv continuous`), so the hosted runner's own speed cancels out.
- **The threshold starts at 10%, not plan.md's 5%,** because the spike's noise between runs of unchanged code
  reached 4% on a quiet machine, and hosted runners share hardware. QS-1105 measures the runners' noise: if
  two runs of the same commit agree within 2.5%, the threshold drops to 5%, as plan.md asks.
- A regression above the threshold is rerun once before it opens a `perf-regression` issue.
- asv runs on the CPU only, on hosted runners. GPU benchmarks run by hand (ADR-0007 D5, amended).

## Consequences

- **Positive:** every published number can be rerun with one command and checked against its machine record;
  JAX's compilation is visible, not hidden or mixed in; a reader is told when a difference is noise.
- **Positive:** a library that gives wrong answers can't win a benchmark.
- **Negative:** benchmarks take longer: 3 processes × 15 samples × at least 0.2 s is at least 9 s per case.
  A full suite of about 100 cases takes 15 to 30 minutes; a quick mode with 1 process and 5 samples is for
  development only and isn't published.
- **Negative:** a 10% regression threshold lets a slowdown of 5 to 10% through until QS-1105 shows the runners
  are quiet enough for 5%. This is a change from plan.md §6.1, approved by the PO in #155.
- **Follow-ups:** QS-1102 builds the harness and the suite; QS-1103 adds the comparison lock group; QS-1105
  measures hosted-runner noise and sets the threshold.

## Validation

- QS-1102's harness has tests that check D1 to D3: a fake slow first call is reported apart; a sample lasts
  at least 0.2 s; each case runs in its own process; a result file without a machine record is rejected.
- Rerunning the headline cases on another day gives medians within 5% of the published ones on the CPU,
  and within 5% on the GPU for calls over 10 ms. If not, D3's repeat count goes up.
- QS-1105 reports two asv runs of the same commit on hosted runners, and the threshold follows D7.
