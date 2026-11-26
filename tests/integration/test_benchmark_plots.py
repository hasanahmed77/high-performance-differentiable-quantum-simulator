"""The plot command redraws every figure from the JSON; the committed ones are current (QS-1104).

These need matplotlib, from the ``bench`` group; without it they're skipped.
"""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

from benchmarks import plots
from benchmarks.run import ROOT


def _case(benchmark: str, runner: str, n: int, median_s: float) -> dict[str, Any]:
    return {
        "id": f"{benchmark}/{runner}/cpu/complex128/n{n}",
        "benchmark": benchmark,
        "runner": runner,
        "device": "cpu",
        "precision": "complex128",
        "n_qubits": n,
        "median_s": median_s,
        "q1_s": median_s * 0.98,
        "q3_s": median_s * 1.03,
        "min_s": median_s * 0.95,
        "first_call_s": median_s * (50 if "jax" in runner else 1.1),
        "processes": [],
    }


def _fixture(tmp_path: Path, name: str, runners: list[str]) -> Path:
    real = json.loads(
        (ROOT / "benchmarks" / "results" / "2026-11-24-cpu-complex128.json").read_text("utf-8")
    )
    cases = [
        _case(b, r, n, 1e-4 * n * (i + 1))
        for b in ("qft", "vqe-grad")
        for i, r in enumerate(runners)
        for n in (4, 8)
        if not (r == "qiskit-aer" and b == "vqe-grad")
    ]
    data = {**real, "cases": cases, "failed": []}
    path = tmp_path / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_every_figure_is_drawn_and_deterministic(tmp_path: Path) -> None:
    _fixture(tmp_path, "with-libraries", ["ketjet-numpy", "ketjet-jax", "qulacs", "qiskit-aer"])
    _fixture(tmp_path, "ketjet-only", ["ketjet-numpy", "ketjet-jax"])
    (tmp_path / "results" / "unpublished-x.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "figures"
    args = ["--results", str(tmp_path / "results"), "--out", str(out)]
    assert plots.main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "ketjet-only-first-call.svg",
        "ketjet-only-gradients.svg",
        "ketjet-only-time.svg",
        "with-libraries-first-call.svg",
        "with-libraries-gradients.svg",
        "with-libraries-speedup.svg",
        "with-libraries-time.svg",
    ]
    svg = (out / "with-libraries-time.svg").read_text(encoding="utf-8")
    assert svg.startswith("<?xml")
    assert "qulacs" in svg
    assert "vqe-grad" in svg
    assert plots.main([*args, "--check"]) == 0
    (out / "ketjet-only-time.svg").write_text("stale", encoding="utf-8")
    assert plots.main([*args, "--check"]) == 1


def test_the_committed_figures_match_the_committed_results() -> None:
    """QS-1104: every figure in the docs is what the committed JSON draws."""
    assert plots.main(["--check"]) == 0


def test_every_runner_has_its_own_colour_marker_and_line() -> None:
    assert set(plots.STYLE) == set(plots.RUNNER_ORDER)
    assert len({colour for colour, _, _ in plots.STYLE.values()}) == len(plots.STYLE)
    assert len({marker for _, marker, _ in plots.STYLE.values()}) == len(plots.STYLE)
