"""The comparison tables pick ketjet's fastest runner and list its losses (QS-1103)."""

from typing import Any

from benchmarks.compare import losses, rows, table


def _case(benchmark: str, runner: str, n: int, median_s: float) -> dict[str, Any]:
    return {"benchmark": benchmark, "runner": runner, "n_qubits": n, "median_s": median_s}


CASES = [
    _case("qft", "ketjet-numpy", 4, 0.004),
    _case("qft", "ketjet-jax", 4, 0.001),
    _case("qft", "qulacs", 4, 0.0005),
    _case("qft", "ketjet-numpy", 8, 0.020),
    _case("qft", "ketjet-jax", 8, 0.010),
    _case("qft", "qulacs", 8, 0.005),
    _case("qft", "qiskit-aer", 8, 0.0105),
    _case("qft", "pennylane-default", 8, 0.100),
    _case("vqe-grad", "ketjet-jax-adjoint", 4, 2.0),
    _case("vqe-grad", "pennylane-lightning", 4, 1.0),
]


def test_rows_use_the_largest_size_and_the_fastest_ketjet_runner() -> None:
    found = rows(CASES)
    assert [(r.benchmark, r.n_qubits, r.ketjet_runner) for r in found] == [
        ("qft", 8, "ketjet-jax"),
        ("vqe-grad", 4, "ketjet-jax-adjoint"),
    ]
    assert found[0].ratio("pennylane-default") == 10.0
    assert len(rows(CASES, all_sizes=True)) == 3


def test_losses_are_listed_worst_first_and_ties_are_not_losses() -> None:
    lost = losses(rows(CASES))
    assert [(row.benchmark, library, ratio) for row, library, ratio in lost] == [
        ("qft", "qulacs", 0.5),
        ("vqe-grad", "pennylane-lightning", 0.5),
    ]


def test_the_table_has_every_verdict() -> None:
    text = table(rows(CASES))
    assert "| `qft` | 8 | 10.0 (ketjet-jax) |" in text
    assert "ketjet 2.00x slower" in text
    assert "ketjet no clear difference (1.05x)" in text
    assert "ketjet 10.00x faster" in text
    # Qiskit Aer has no gradients, so its vqe-grad cell is empty.
    assert text.splitlines()[-1].count("|  |") == 2
