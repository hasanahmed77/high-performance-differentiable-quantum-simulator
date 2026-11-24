"""The "Running on a GPU" section of Getting started runs, on the CPU too (QS-607).

The example is written to run anywhere: without a GPU, both of its runs use the CPU. CI has no
GPU, so this test keeps the example correct; ``tests/gpu`` runs the same computations on one.
"""

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")

GUIDE = Path(__file__).parents[2] / "docs" / "getting-started.md"


def _section() -> str:
    text = GUIDE.read_text(encoding="utf-8")
    return text.split("## Running on a GPU\n", 1)[1].split("\n## ", 1)[0]


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _section(), re.S)


def test_the_section_has_one_python_example() -> None:
    assert len(_python_blocks()) == 1  # a new example needs a test here


def test_the_example_runs_and_prints_what_its_comments_say() -> None:
    namespace: dict[str, Any] = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(_python_blocks()[0], namespace)  # our own docs
    assert out.getvalue().splitlines() == [jax.default_backend(), "True"]
    np.testing.assert_allclose(
        np.asarray(namespace["state"]), [np.cos(0.25), 0, 0, np.sin(0.25)], atol=1e-12
    )
    assert namespace["on_cpu"].devices() == {jax.devices("cpu")[0]}


def test_the_install_command_names_jaxs_cuda_build() -> None:
    assert 'pip install "ketjet[jax]" "jax[cuda12]"' in _section()
