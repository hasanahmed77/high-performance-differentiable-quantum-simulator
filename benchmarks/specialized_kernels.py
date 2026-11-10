"""Benchmark: specialized diagonal and permutation kernels vs the dense kernel (QS-205).

Run with ``uv run python benchmarks/specialized_kernels.py``. It prints a Markdown table of the
best-of-``REPEATS`` time per gate application, as the backend applies it (matrix construction
included), and the speedup over ``apply_matrix``. The asv suite (QS-1102) will replace this.
"""

import platform
import timeit
from collections.abc import Callable

import numpy as np

from ketjet import operations as ops
from ketjet.backends import NumPyBackend
from ketjet.kernels import apply_matrix
from ketjet.operations import Operation
from ketjet.statevector import StateVector as State

REPEATS = 7
SIZES = (10, 16, 20, 22)


def _gates(n: int) -> list[tuple[str, Operation]]:
    mid = n // 2
    return [
        ("RZ (diagonal)", ops.RZ(0.3, mid)),
        ("T (diagonal)", ops.T(n - 1)),
        ("CZ (diagonal)", ops.CZ(0, n - 1)),
        ("X (permutation)", ops.X(mid)),
        ("CNOT (permutation)", ops.CNOT(0, n - 1)),
        ("SWAP (permutation)", ops.SWAP(1, n - 2)),
        ("Toffoli (permutation)", ops.Toffoli(0, mid, n - 1)),
    ]


def _best(fn: Callable[[], object], number: int) -> float:
    return min(timeit.repeat(fn, number=number, repeat=REPEATS)) / number


def main() -> None:
    """Print the benchmark table."""
    backend = NumPyBackend()
    print(f"numpy {np.__version__}, Python {platform.python_version()}, {platform.machine()}")
    print(f"{platform.platform()}; best of {REPEATS}, complex128\n")
    print("| Gate | Qubits | Dense | Specialized | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for n in SIZES:
        rng = np.random.default_rng(n)
        state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        number = max(1, 2**18 // 2**n)
        for label, op in _gates(n):

            def dense(op: Operation = op, state: State = state) -> State:
                return apply_matrix(state, op.matrix(state.dtype), op.all_wires)

            def fast(op: Operation = op, state: State = state) -> State:
                return backend.apply_operation(state, op)

            np.testing.assert_allclose(fast(), dense(), atol=1e-12)
            t_dense, t_fast = _best(dense, number), _best(fast, number)
            print(
                f"| {label} | {n} | {t_dense * 1e3:.3f} ms | {t_fast * 1e3:.3f} ms "
                f"| {t_dense / t_fast:.1f}x |"
            )


if __name__ == "__main__":
    main()
