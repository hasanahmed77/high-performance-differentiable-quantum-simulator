"""Every benchmark still runs and checks its answer (QS-1102), so the suite can't rot.

CI runs each benchmark at a tiny size, in this process, with one sample. The published numbers
come from ``python -m benchmarks.run`` on the workstation (ADR-0008).
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
from benchmarks import harness
from benchmarks.cases import SMOKE_SIZE, Case, WrongResultError, cases, prepare
from benchmarks.circuits import BENCHMARKS, GRADIENT_BENCHMARKS, ising, qft
from benchmarks.run import main, run_case

from ketjet.observables import Sum


@pytest.fixture(autouse=True)
def _restore_x64() -> Iterator[None]:
    """The harness sets JAX's 64-bit mode as a user would; put it back for the other tests."""
    x64 = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", x64)  # type: ignore[no-untyped-call]


@pytest.mark.parametrize("case", cases(sizes=[SMOKE_SIZE]), ids=lambda c: c.id)
def test_every_benchmark_runs_and_matches_the_oracle(case: Case) -> None:
    result = run_case(case, harness.SMOKE)
    assert result["first_call_s"] > 0
    assert len(result["samples"]) == 1
    assert result["info"]["gates"] > 0


@pytest.mark.parametrize(
    "case",
    [c for c in cases(sizes=[SMOKE_SIZE], precision="complex64") if c.runner != "ketjet-numpy"],
    ids=lambda c: c.id,
)
def test_jax_runners_in_complex64(case: Case) -> None:
    result = run_case(case, harness.SMOKE)
    assert result["info"]["jax_x64"] is False


def test_a_wrong_answer_is_refused() -> None:
    """ADR-0008 D1: a runner that gives a wrong answer isn't timed."""
    prepared = prepare(Case("gate-h", "ketjet-numpy", SMOKE_SIZE))
    right = np.asarray(prepared.call())
    prepared.check(right)
    with pytest.raises(WrongResultError, match="disagrees with the oracle"):
        prepared.check(right[::-1])


def test_case_selection() -> None:
    every = cases()
    assert {c.benchmark for c in every if c.runner == "ketjet-jax-adjoint"} == GRADIENT_BENCHMARKS
    assert {c.runner for c in cases(device="gpu")} == {"ketjet-jax", "ketjet-jax-adjoint"}
    assert len({c.id for c in every}) == len(every)
    with pytest.raises(ValueError, match="Unknown benchmark"):
        cases(["nope"])
    with pytest.raises(ValueError, match="Unknown runner"):
        cases(runners=["qiskit"])


def test_a_gpu_case_refuses_to_run_on_the_cpu() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("a GPU is present")
    with pytest.raises(RuntimeError, match="needs JAX on the gpu"):
        prepare(Case("qft", "ketjet-jax", SMOKE_SIZE, device="gpu"))


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmark_circuits_are_seeded(name: str) -> None:
    """Every run, and every library, gets the same circuit and angles (ADR-0008 D5)."""
    a, b = BENCHMARKS[name](5), BENCHMARKS[name](5)
    assert a.circuit == b.circuit
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("basis", [0, 1, 0b0110, 0b1011])
def test_qft_is_the_fourier_transform(basis: int) -> None:
    n = 4
    spec = qft(n)
    bits = [(basis >> (n - 1 - w)) & 1 for w in range(n)]  # wire 0 is the most significant
    state = spec.circuit.bind({f"p{w}": np.pi * b for w, b in enumerate(bits)}).state()
    size = 2**n
    expected = np.exp(2j * np.pi * basis * np.arange(size) / size) / np.sqrt(size)
    np.testing.assert_allclose(state, expected, atol=1e-12)


def test_ising_has_every_term() -> None:
    hamiltonian = ising(4)
    assert isinstance(hamiltonian, Sum)
    assert len(hamiltonian.terms) == 3 + 4


def test_the_runner_writes_a_valid_file_from_separate_processes(tmp_path: Path) -> None:
    """ADR-0008 D3 and D6: each case in its own process; the file carries its machine record."""
    out = tmp_path / "result.json"
    code = main(
        [
            "--benchmarks",
            "gate-h,vqe-grad",
            "--runners",
            "ketjet-numpy",
            "--smoke",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    harness.validate(data)
    assert data["settings"]["published"] is False
    pids = [p["pid"] for case in data["cases"] for p in case["processes"]]
    assert len(pids) == len(set(pids)) == 2
    assert os.getpid() not in pids
    assert [c["n_qubits"] for c in data["cases"]] == [SMOKE_SIZE, SMOKE_SIZE]
