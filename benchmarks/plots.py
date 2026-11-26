"""Draw every benchmark figure from the committed result files (QS-1104).

Examples::

    uv sync --group bench
    uv run python -m benchmarks.plots                 # every file in benchmarks/results/
    uv run python -m benchmarks.plots --check         # fail if a committed figure is out of date

Each result file gets its own figures, named after it, in ``docs/assets/benchmarks/``: numbers
from two machines or two runs are never drawn in one plot (ADR-0008 D6). The figures are:

``<file>-time.svg``
    Steady-state median against qubits for every benchmark and runner, with the interquartile
    range, on log axes (ADR-0008 D4).
``<file>-speedup.svg``
    Each library's time over ketjet's fastest runner's, against qubits; the grey band is "no
    clear difference" (0.9 to 1.1). Only for files with other libraries in them.
``<file>-gradients.svg``
    The cost of a gradient against qubits, by method: ketjet's adjoint (NumPy and JAX) and
    backprop (JAX), and each library's own.
``<file>-first-call.svg``
    The first call, which for JAX includes compilation, against the steady state.

Unpublished files (``unpublished-*.json``) are skipped. The SVGs are deterministic, so
``--check`` can tell whether they match the JSON.
"""

import argparse
import io
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from benchmarks import harness
from benchmarks.cases import KETJET_RUNNERS
from benchmarks.circuits import BENCHMARKS, GRADIENT_BENCHMARKS
from benchmarks.libraries import LIBRARY_RUNNERS

ROOT: Final = Path(__file__).resolve().parent.parent
RESULTS: Final = ROOT / "benchmarks" / "results"
FIGURES: Final = ROOT / "docs" / "assets" / "benchmarks"

SURFACE: Final = "#fcfcfb"
TEXT: Final = "#0b0b0b"
TEXT_SECONDARY: Final = "#52514e"
GRID: Final = "#ecebe7"
BAND: Final = "#e4e3df"

STYLE: Final[dict[str, tuple[str, str, str]]] = {
    # runner: (colour, marker, line style). The colours are the validated categorical order, one
    # per runner and never reassigned; ketjet's lines are solid, the other libraries' dashed, and
    # each has its own marker, so a line is never identified by colour alone.
    "ketjet-numpy": ("#2a78d6", "o", "-"),
    "ketjet-jax": ("#eb6834", "s", "-"),
    "ketjet-jax-adjoint": ("#1baf7a", "D", "-"),
    "pennylane-default": ("#eda100", "v", "--"),
    "pennylane-lightning": ("#e87ba4", "^", "--"),
    "qiskit-aer": ("#008300", "P", "--"),
    "qulacs": ("#6250d6", "X", "--"),
}
"""Every runner's colour, marker and line style."""

RUNNER_ORDER: Final = (*KETJET_RUNNERS, *LIBRARY_RUNNERS)


def _pyplot() -> Any:
    import matplotlib as mpl  # noqa: PLC0415 - only the plots need matplotlib

    mpl.use("svg")
    mpl.rcParams.update(
        {
            "svg.hashsalt": "ketjet",  # deterministic ids, so --check can compare files
            "svg.fonttype": "none",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
    )
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def _series(
    cases: Iterable[Mapping[str, Any]], benchmark: str
) -> dict[str, list[Mapping[str, Any]]]:
    by_runner: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        if case["benchmark"] == benchmark:
            by_runner.setdefault(case["runner"], []).append(case)
    ordered = {r: by_runner[r] for r in RUNNER_ORDER if r in by_runner}
    return {r: sorted(cs, key=lambda c: c["n_qubits"]) for r, cs in ordered.items()}


def _line(ax: Any, runner: str, xs: Sequence[float], ys: Sequence[float], **kwargs: Any) -> None:
    colour, marker, style = STYLE[runner]
    ax.plot(xs, ys, color=colour, marker=marker, linestyle=style, label=runner, **kwargs)


def _qubit_axis(ax: Any, cases: Iterable[Mapping[str, Any]]) -> None:
    ax.set_xticks(sorted({c["n_qubits"] for c in cases}))
    ax.set_xlabel("qubits")


def _legend(fig: Any, axes: Sequence[Any]) -> None:
    handles: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    labels = [r for r in RUNNER_ORDER if r in handles]
    columns = min(len(labels), 2 * sum(ax.get_visible() for ax in fig.axes[:4]))
    fig.legend(
        [handles[r] for r in labels],
        labels,
        loc="lower center",
        ncol=columns,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.0),
    )
    rows = -(-len(labels) // columns)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height + 0.25 * rows)
    fig.legend_bottom = (0.25 * rows + 0.15) / (height + 0.25 * rows)


def _grid(plt: Any, panels: int, title: str) -> tuple[Any, list[Any]]:
    columns = min(panels, 4)
    rows = -(-panels // columns)
    fig, grid = plt.subplots(
        rows, columns, figsize=(2.9 * columns, 2.5 * rows + 0.5), squeeze=False
    )
    axes = [ax for row in grid for ax in row]
    for ax in axes[panels:]:
        ax.set_visible(False)
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig, axes[:panels]


def _benchmarks(cases: Sequence[Mapping[str, Any]], only: Iterable[str] = BENCHMARKS) -> list[str]:
    present = {c["benchmark"] for c in cases}
    return [b for b in only if b in present]


def time_figure(plt: Any, data: Mapping[str, Any], name: str) -> Any:
    """Draw the median time against qubits, one panel per benchmark, with IQR bars."""
    cases = data["cases"]
    names = _benchmarks(cases)
    fig, axes = _grid(plt, len(names), f"Steady-state time, median of 3 processes ({name})")
    for ax, benchmark in zip(axes, names, strict=True):
        for runner, series in _series(cases, benchmark).items():
            xs = [c["n_qubits"] for c in series]
            ms = [c["median_s"] * 1e3 for c in series]
            low = [(c["median_s"] - c["q1_s"]) * 1e3 for c in series]
            high = [(c["q3_s"] - c["median_s"]) * 1e3 for c in series]
            _line(ax, runner, xs, ms)
            ax.errorbar(xs, ms, yerr=[low, high], fmt="none", ecolor=STYLE[runner][0], lw=1)
        ax.set_yscale("log")
        ax.set_title(benchmark, loc="left")
        ax.set_ylabel("ms")
        _qubit_axis(ax, [c for c in cases if c["benchmark"] == benchmark])
    _legend(fig, axes)
    return fig


def speedup_figure(plt: Any, data: Mapping[str, Any], name: str) -> Any | None:
    """Draw each library's time over ketjet's fastest runner's; above 1, ketjet is faster."""
    cases = data["cases"]
    if not any(c["runner"] in LIBRARY_RUNNERS for c in cases):
        return None
    names = _benchmarks(cases)
    fig, axes = _grid(
        plt,
        len(names),
        f"Library time / ketjet's fastest runner: above 1, ketjet is faster ({name})",
    )
    low, high = harness.NO_CLEAR_DIFFERENCE
    for ax, benchmark in zip(axes, names, strict=True):
        series = _series(cases, benchmark)
        best: dict[int, float] = {}
        for runner in KETJET_RUNNERS:
            for case in series.get(runner, []):
                n = case["n_qubits"]
                best[n] = min(case["median_s"], best.get(n, case["median_s"]))
        ax.axhspan(low, high, color=BAND, lw=0)
        ax.axhline(1, color=TEXT_SECONDARY, lw=0.8)
        for runner in LIBRARY_RUNNERS:
            points = [(c["n_qubits"], c["median_s"]) for c in series.get(runner, [])]
            points = [(n, t / best[n]) for n, t in points if n in best]
            if points:
                _line(ax, runner, *zip(*points, strict=True))
        ax.set_yscale("log")
        ax.set_title(benchmark, loc="left")
        ax.set_ylabel("ratio")
        _qubit_axis(ax, [c for c in cases if c["benchmark"] == benchmark])
    _legend(fig, axes)
    return fig


def gradients_figure(plt: Any, data: Mapping[str, Any], name: str) -> Any | None:
    """Draw the gradient benchmarks' times by method, one panel per benchmark."""
    cases = data["cases"]
    names = _benchmarks(cases, sorted(GRADIENT_BENCHMARKS, key=list(BENCHMARKS).index))
    if not names:
        return None
    fig, axes = _grid(plt, len(names), f"Gradient time by method ({name})")
    for ax, benchmark in zip(axes, names, strict=True):
        for runner, series in _series(cases, benchmark).items():
            _line(
                ax, runner, [c["n_qubits"] for c in series], [c["median_s"] * 1e3 for c in series]
            )
        ax.set_yscale("log")
        ax.set_title(benchmark, loc="left")
        ax.set_ylabel("ms per gradient")
        _qubit_axis(ax, [c for c in cases if c["benchmark"] == benchmark])
    _legend(fig, axes)
    return fig


def first_call_figure(plt: Any, data: Mapping[str, Any], name: str) -> Any:
    """Draw the first call over the steady state: what a single call costs on top."""
    cases = data["cases"]
    names = _benchmarks(cases)
    fig, axes = _grid(
        plt, len(names), f"First call / steady state: JAX's includes compilation ({name})"
    )
    for ax, benchmark in zip(axes, names, strict=True):
        for runner, series in _series(cases, benchmark).items():
            xs = [c["n_qubits"] for c in series]
            _line(ax, runner, xs, [c["first_call_s"] / c["median_s"] for c in series])
        ax.axhline(1, color=TEXT_SECONDARY, lw=0.8)
        ax.set_yscale("log")
        ax.set_title(benchmark, loc="left")
        ax.set_ylabel("ratio")
        _qubit_axis(ax, [c for c in cases if c["benchmark"] == benchmark])
    _legend(fig, axes)
    return fig


FIGURES_BY_NAME: Final = {
    "time": time_figure,
    "speedup": speedup_figure,
    "gradients": gradients_figure,
    "first-call": first_call_figure,
}


def render(data: Mapping[str, Any], name: str) -> dict[str, str]:
    """Return every figure for one result file, as ``{file name: SVG text}``."""
    plt = _pyplot()
    out = {}
    for kind, draw in FIGURES_BY_NAME.items():
        fig = draw(plt, data, name)
        if fig is None:
            continue
        fig.tight_layout(rect=(0, fig.legend_bottom, 1, 1))
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        plt.close(fig)
        out[f"{name}-{kind}.svg"] = buffer.getvalue()
    return out


def result_files(results: Path) -> list[Path]:
    """Return the published result files in ``results``, oldest first."""
    return sorted(p for p in results.glob("*.json") if not p.name.startswith("unpublished-"))


def main(argv: Sequence[str] | None = None) -> int:
    """Draw (or, with ``--check``, compare) every figure."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.plots", description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=FIGURES)
    parser.add_argument("--check", action="store_true", help="fail if a figure is out of date")
    args = parser.parse_args(argv)
    stale = []
    count = 0
    for path in result_files(args.results):
        data = json.loads(path.read_text(encoding="utf-8"))
        harness.validate(data)
        for file_name, svg in render(data, path.stem).items():
            count += 1
            target = args.out / file_name
            if args.check:
                if not target.exists() or target.read_text(encoding="utf-8") != svg:
                    stale.append(file_name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(svg, encoding="utf-8", newline="\n")
    if stale:
        print("Out of date; run `uv run python -m benchmarks.plots`:\n  " + "\n  ".join(stale))
        return 1
    print(f"{'Checked' if args.check else 'Wrote'} {count} figures in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
