"""Tables of ketjet against the other simulators, from a result file (QS-1103).

Examples::

    uv run python -m benchmarks.compare benchmarks/results/2026-11-26-cpu-complex128.json
    uv run python -m benchmarks.compare <file> --all-sizes

For each benchmark and size, ketjet's time is its fastest runner's steady-state median, and each
library's is divided by it (ADR-0008 D4): above 1.1 ketjet is faster, below 0.9 it is slower, and
in between there's no clear difference. Losses are listed, not left out (plan.md G3).
"""

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks import harness
from benchmarks.cases import KETJET_RUNNERS
from benchmarks.libraries import LIBRARY_RUNNERS


@dataclass(frozen=True)
class Row:
    """One benchmark at one size: ketjet's best time and each library's."""

    benchmark: str
    n_qubits: int
    ketjet_runner: str
    ketjet_s: float
    library_s: dict[str, float]

    def ratio(self, library: str) -> float:
        """Return the library's time over ketjet's: above 1, ketjet is faster."""
        return self.library_s[library] / self.ketjet_s


def rows(cases: Iterable[Mapping[str, Any]], all_sizes: bool = False) -> list[Row]:
    """Group ``cases`` into rows, at each benchmark's largest size unless ``all_sizes``."""
    times: dict[tuple[str, int], dict[str, float]] = {}
    for case in cases:
        times.setdefault((case["benchmark"], case["n_qubits"]), {})[case["runner"]] = case[
            "median_s"
        ]
    largest: dict[str, int] = {}
    for benchmark, n in times:
        largest[benchmark] = max(n, largest.get(benchmark, n))
    result = []
    for (benchmark, n), by_runner in sorted(times.items()):
        if not all_sizes and n != largest[benchmark]:
            continue
        ketjet = {r: t for r, t in by_runner.items() if r in KETJET_RUNNERS}
        if not ketjet:
            continue
        best = min(ketjet, key=ketjet.__getitem__)
        libraries = {r: t for r, t in by_runner.items() if r in LIBRARY_RUNNERS}
        result.append(Row(benchmark, n, best, ketjet[best], libraries))
    return result


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:,.1f}" if seconds < 1 else f"{seconds * 1e3:,.0f}"


def table(table_rows: Sequence[Row]) -> str:
    """Return a Markdown table: ketjet's best time, then each library's time and verdict."""
    libraries = [r for r in LIBRARY_RUNNERS if any(r in row.library_s for row in table_rows)]
    header = [
        "Benchmark",
        "Qubits",
        "ketjet (fastest runner), ms",
        *(f"`{r}`, ms" for r in libraries),
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table_rows:
        cells = [
            f"`{row.benchmark}`",
            str(row.n_qubits),
            f"{_ms(row.ketjet_s)} ({row.ketjet_runner})",
        ]
        for library in libraries:
            if library not in row.library_s:
                cells.append("")
                continue
            verdict = harness.compare(row.ketjet_s, row.library_s[library])
            cells.append(f"{_ms(row.library_s[library])}: ketjet {verdict}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def losses(table_rows: Sequence[Row]) -> list[tuple[Row, str, float]]:
    """Return every ``(row, library, ratio)`` where ketjet is clearly slower, worst first."""
    low, _ = harness.NO_CLEAR_DIFFERENCE
    found = [
        (row, library, row.ratio(library))
        for row in table_rows
        for library in row.library_s
        if row.ratio(library) < low
    ]
    return sorted(found, key=lambda item: item[2])


def main(argv: Sequence[str] | None = None) -> int:
    """Print the comparison table and ketjet's losses for a result file."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.compare", description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--all-sizes", action="store_true", help="every size, not the largest")
    args = parser.parse_args(argv)
    data = json.loads(args.result.read_text(encoding="utf-8"))
    harness.validate(data)
    found = rows(data["cases"], args.all_sizes)
    print(table(found))
    lost = losses(found)
    print(f"\nketjet is clearly slower in {len(lost)} of {sum(len(r.library_s) for r in found)}:")
    for row, library, ratio in lost:
        print(f"- `{row.benchmark}`, {row.n_qubits} qubits: {library} is {1 / ratio:.1f}x faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
