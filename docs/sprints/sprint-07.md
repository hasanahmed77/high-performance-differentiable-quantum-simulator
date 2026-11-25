# Sprint 07: Benchmarks and the first performance report

| | |
|---|---|
| **Status** | Planned, waiting for the PO's approval in this PR |
| **Started** | 2026-11-24 |
| **Milestone** | `Sprint 07 – Benchmarks and first performance report` |
| **Sprint goal** | Honest, reproducible numbers against the competition: one command runs a benchmark suite under ADR-0008, compares ketjet with `default.qubit`, `lightning.qubit`, Qiskit Aer and Qulacs on the same circuits and the same machine, and regenerates every plot, including the ones where ketjet loses. A nightly asv job catches regressions. |

## Sprint backlog

| ID | Story | Pts | Depends on | PR |
|---|---|---|---|---|
| QS-1102 | Benchmark suite and harness: single-gate throughput, QFT, random circuits, a VQE ansatz with gradients, batched QML; JSON results committed | 5 | ADR-0008 | #161 |
| QS-1103 | Comparisons with `default.qubit`, `lightning.qubit`, Qiskit Aer and Qulacs on the CPU; `lightning.gpu` and Aer on the GPU, run by hand | 5 | QS-1102 | |
| QS-1104 | Plots regenerated from the JSON by one command: time against qubits, speedup, gradient cost | 3 | QS-1102 | |
| QS-1105 | Nightly asv regression tracking, a published dashboard, and an issue opened on a regression over 10% | 3 | n/a | |
| QS-407 | VQE H₂ example converges to the known energy (re-estimated from 2, see below) | 0 | n/a | |
| QS-1108 | **New, stretch:** how many qubits the JAX backend fits on the RTX 4060, for each gradient method and precision | 2 | QS-1102 | |
| | **Total** | **18** (16 committed + 2 stretch) | | |

**Capacity.** Mean throughput over seven sprints is 20.9 points. The 16 committed points leave room for running the
benchmarks themselves, which takes machine time more than coding time, and for the GPU comparisons by hand.

**Re-estimate.** **QS-407, 2 → 0.** Its acceptance is already met: `tests/integration/test_vqe_h2.py` reaches the
FCI energy of H₂ with adjoint on NumPy and backprop on JAX, and Sprint 6's demo did it for 64 starting points on the
GPU. The story closes with that evidence and no new code.

**New story, QS-1108 (stretch).** The Sprint 6 review found that the GPU adjoint uses about 12.5 states at 18
qubits, so plan.md §4.5's limits (27 qubits in complex64, 26 in complex128, on 8 GB) are unverified. QS-1108 finds
the largest circuit that runs for adjoint and backprop in both precisions, and corrects §4.5 with the measured
numbers.

**Spike (2026-11-24).** `pennylane-lightning` 0.45, `qiskit-aer` 0.18.4 and `qulacs` 0.6.14 install together with
PennyLane 0.45.1 and NumPy 2.5.5 on Python 3.13 in the `ketjet-gpu` image, and give the same expectation value on a
3-qubit test circuit. Qulacs's `RY` turns the opposite way to PennyLane's, so every translated circuit is checked
against ketjet's oracle before it's timed (ADR-0008 D1). `qulacs-gpu` hasn't had a release since 2022, so it's left
out.

**Sequencing.** QS-1102 first: the harness, the circuits and the JSON format that everything else reads. Then
QS-1103 adds the other libraries to the same harness, and QS-1104 draws the plots from its JSON. QS-1105 is
independent and can go in while another PR waits on CI. QS-1108 last, at the workstation. One story PR in review at
a time (Sprint 6 retro).

**Scope changes in this plan (for PO approval):**

- **All benchmarks run on the workstation, in the `ketjet-gpu` image.** plan.md planned a Mac for the CPU numbers,
  and there is no Mac. The JSON records the CPU, GPU, driver, library versions and thread settings (ADR-0008 D6).
  The PC should be otherwise idle while a benchmark runs.
- **The GPU comparisons are run by hand,** like every GPU test (ADR-0007 D5, amended): `lightning.gpu` and
  `qiskit-aer-gpu`. They need NVIDIA's cuQuantum and CUDA libraries, a large download, so they stay out of the
  lockfile and are installed in a throwaway environment in the container, after the PO approves the download.
- **Qulacs on the CPU only,** because `qulacs-gpu` is abandoned.
- **The comparison libraries go in their own `bench` dependency group,** with matplotlib and asv, so they never
  reach users or the normal test run.
- **`make plots` becomes `uv run python -m benchmarks.plots`.** The repository has no Makefile, and Windows has no
  `make`.
- **The regression threshold is 10%, not 5%,** as already approved in ADR-0008 D7. It drops to 5% if hosted-runner
  noise turns out to be 2.5% or less.
- **The asv dashboard is published on the existing docs site** (GitHub Pages), under `/benchmarks/`. Its history
  comes from hosted runners, whose machines vary, so the page says that regressions are judged by `asv continuous`
  within one job, not by the dashboard's line.

## Test plan

| ID | Verification |
|---|---|
| QS-1102 | The harness refuses to time a circuit whose result disagrees with the NumPy oracle; warm-up, sample counts and process isolation follow ADR-0008 D1–D3 (a test reads the harness's settings); the JSON validates against a schema; **can't rot:** CI runs every benchmark at a tiny size on every PR |
| QS-1103 | Every library's result equals the oracle on every benchmark circuit at small sizes, in both precisions where the library has them; the results table lists where ketjet loses, with the ratio; the GPU runs' output is pasted into the PR |
| QS-1104 | The plot command regenerates every figure from the committed JSON; a test runs it on a small fixture |
| QS-1105 | `asv run --quick` in CI on every PR; the nightly workflow runs only on `schedule` and `workflow_dispatch` (a test reads the workflow); a deliberately slowed branch, dispatched by hand, opens an issue; the dashboard is reachable on the docs site |
| QS-407 | The existing `tests/integration/test_vqe_h2.py` |
| QS-1108 | The largest qubit count that fits, for adjoint and backprop in complex64 and complex128, measured on the RTX 4060 and written into plan.md §4.5 |
| Sprint | **Integration:** one command, run on a clean checkout in the image, regenerates the JSON and every plot. **Report:** the first numbers, with the hardware record, on the docs site |

## Demo

The first comparison plots, from one command:

```bash
uv run python -m benchmarks.run --suite all    # writes benchmarks/results/<date>.json
uv run python -m benchmarks.plots              # redraws every figure from the JSON
```

And a table of where ketjet loses. The expected answer is "often, on the CPU": `lightning.qubit` is compiled C++,
and ketjet's own native backend is Sprint 8's work (goal G3). The numbers show by how much, and where JAX's
compilation or batching closes the gap.

## Actions needed from the PO

1. **Approve this plan,** the re-estimate, the new QS-1108 and the scope changes above, by merging this PR.
2. **Let the nightly asv job write to the repository.** It needs `issues: write` to open a regression issue, and
   `contents: write` to keep its result history on a separate `asv-results` branch. Without the second, the
   dashboard shows only the latest run.
3. **Approve the GPU download when QS-1103 gets there.** I'll give you the exact size before downloading.
4. **Decide on the Sprint 6 retro's open item:** should CI automatically retry jobs that GitHub never started?
5. **Approve Dependabot's #156** (numpy 2.5.5, ruff 0.17.5, diff-cover 11.0.2; lockfile only; its checks passed).
