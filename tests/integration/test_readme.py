"""The README's examples run and print what the README says they print (QS-1203).

The README is also the PyPI page, so its examples are the first code most users try.
"""

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import pytest

qml = pytest.importorskip("pennylane")  # the quickstart uses the PennyLane device
pytestmark = pytest.mark.pennylane

README = Path(__file__).parents[2] / "README.md"
REPO = "https://github.com/hasanahmed77/high-performance-differentiable-quantum-simulator"


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def _run(block: str) -> tuple[dict[str, Any], list[str]]:
    namespace: dict[str, Any] = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(block, namespace)  # our own README
    return namespace, out.getvalue().splitlines()


@pytest.mark.parametrize("index", range(2))
def test_every_example_prints_what_its_comments_say(index: int) -> None:
    """Each print(...) line carries a comment with its exact output."""
    block = _blocks()[index]
    expected = [
        line.split("#", 1)[1].strip().split(",")[0].strip()
        for line in block.splitlines()
        if line.startswith("print(")
    ]
    _, printed = _run(block)
    assert len(printed) == len(expected)
    for got, want in zip(printed, expected, strict=True):
        assert got.strip() == want


def test_the_readme_has_exactly_the_tested_examples() -> None:
    assert len(_blocks()) == 2  # a new example needs a new case above


def test_the_quickstart_reaches_the_exact_minimum() -> None:
    namespace, _ = _run(_blocks()[0])
    exact = min(
        qml.eigvals(qml.PauliZ(0) @ qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1)).real
    )
    assert abs(namespace["energy"](namespace["theta"]) - exact) < 1e-8


def test_links_work_on_pypi() -> None:
    """PyPI renders the README without the repository, so relative links would break."""
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        assert target.startswith(("https://", "#")), target
    assert f"{REPO}/blob/main/LICENSE" in text
