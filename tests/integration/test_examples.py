"""The example scripts in ``examples/`` run and reach their results (QS-1202).

Each script asserts its own result (an exact probability, the QFT formula, the FCI energy), so a
wrong answer fails here, not only a crash. They run as subprocesses, exactly as a user would
run them.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pennylane")
pytestmark = pytest.mark.pennylane

EXAMPLES = Path(__file__).parents[2] / "examples"
EXPECTED = {
    "bell.py": ["probabilities |00>, |01>, |10>, |11>: [0.5 0.  0.  0.5]", "<Z0 Z1> = 1.000"],
    "qft.py": ["QFT|0110> on 4 qubits"],
    "vqe.py": ["exact (FCI) energy:  -1.13726973 Ha", "VQE energy:          -1.13726973 Ha"],
}


def test_every_example_is_tested() -> None:
    assert sorted(p.name for p in EXAMPLES.glob("*.py")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_runs_and_reaches_its_result(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for line in EXPECTED[name]:
        assert line in result.stdout, result.stdout
